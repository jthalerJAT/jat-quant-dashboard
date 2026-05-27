// Vercel serverless function: scrape a Twitter handle's last 90 days,
// extract trading ideas via batched Claude call, return JSON.
//
// GET  /api/scrape-handle?handle=ben_bajarin
// POST /api/scrape-handle  body: { "handle": "ben_bajarin" }
//
// Required env vars:
//   ANTHROPIC_API_KEY     — Anthropic API key
//   TWITTER_BEARER_TOKEN  — X API v2 Bearer (Pro tier so 90d lookback + user timeline works)
//
// Vercel Pro plan required for the 60s maxDuration (set in vercel.json).
//
// Wall-clock budget design (60s total Vercel cap → 55s usable):
//   - X user lookup     : 1 call, ~0.5s
//   - X timeline fetch  : up to MAX_PAGES, bounded by X_FETCH_BUDGET_MS
//   - Anthropic batch   : bounded by ANTHROPIC_TIMEOUT_MS
// If the X fetch is throttled or slow, we return whatever we have plus a
// `truncated: true` flag so the UI can warn the user instead of getting a 504.

import Anthropic from "@anthropic-ai/sdk";

export const config = {
  maxDuration: 60,
};

const HANDLE_RE = /^[A-Za-z0-9_]{1,15}$/;
const NINETY_DAYS_MS = 90 * 24 * 60 * 60 * 1000;

const MAX_PAGES = 4;                  // 400 tweets ceiling
const PER_X_CALL_TIMEOUT_MS = 7000;   // 7s per X HTTP request
const X_FETCH_BUDGET_MS = 15000;      // total wall-clock cap for all X calls

// Anthropic: parallel chunking instead of one giant call.
// Single Claude call on 300+ tweets reliably blew >30s (504 on jukan05).
// Two parallel calls of ~half the tweets each finish in ~half the wall time.
const ANTHROPIC_CHUNK_SIZE = 175;     // tweets per parallel call
const ANTHROPIC_TIMEOUT_MS = 30000;   // per-call SDK timeout
const ANTHROPIC_WALL_MS = 34000;      // hard outer Promise.race cap per chunk
const ANTHROPIC_MAX_RETRIES = 0;      // default 2 — would blow our budget on retry

const EXTRACTION_SCHEMA = {
  type: "object",
  properties: {
    ideas: {
      type: "array",
      description:
        "Trading ideas extracted from the tweets. One entry per tweet that contains a specific, actionable, directional thesis on a publicly-traded ticker. Skip generic market commentary, retweets-without-comment, and tweets without a clear directional view. Sort by date descending. Limit to 50.",
      items: {
        type: "object",
        properties: {
          tweet_id: { type: "string", description: "The tweet's id as given in the [bracketed id] prefix." },
          date: { type: "string", description: "The tweet's created_at timestamp (ISO 8601)." },
          ticker: { type: "string", description: "Uppercase ticker symbol (e.g., NVDA). Omit the entry entirely if no ticker is mentioned." },
          stance: { type: "string", description: "One of: bullish, bearish, neutral." },
          summary: { type: "string", description: "1–2 sentence summary of the thesis. Be specific — include price targets, catalysts, and timeframe when present." },
        },
        required: ["tweet_id", "date", "ticker", "stance", "summary"],
        additionalProperties: false,
      },
    },
  },
  required: ["ideas"],
  additionalProperties: false,
};

function jsonResponse(res, status, body) {
  res.status(status).setHeader("Content-Type", "application/json");
  res.send(JSON.stringify(body));
}

function fetchWithTimeout(url, opts, timeoutMs) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(new Error("timeout")), timeoutMs);
  return fetch(url, { ...(opts || {}), signal: ac.signal }).finally(() => clearTimeout(t));
}

async function lookupUser(handle, bearer) {
  const r = await fetchWithTimeout(
    `https://api.twitter.com/2/users/by/username/${handle}`,
    { headers: { Authorization: `Bearer ${bearer}` } },
    PER_X_CALL_TIMEOUT_MS,
  );
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    const err = new Error(`X user lookup failed (${r.status})`);
    err.status = r.status === 404 ? 404 : 502;
    err.detail = detail;
    throw err;
  }
  const body = await r.json();
  if (!body.data?.id) {
    const err = new Error("Handle not found on X");
    err.status = 404;
    throw err;
  }
  return body.data;
}

async function fetchTimeline(userId, bearer, startIso, budgetDeadlineMs) {
  const out = [];
  let nextToken = null;
  let pagesAttempted = 0;
  let truncated = false;

  for (let i = 0; i < MAX_PAGES; i++) {
    pagesAttempted++;
    if (Date.now() > budgetDeadlineMs) {
      console.log(`[scrape-handle] X budget exhausted after ${pagesAttempted - 1} pages`);
      truncated = true;
      break;
    }
    const params = new URLSearchParams({
      max_results: "100",
      start_time: startIso,
      "tweet.fields": "created_at,public_metrics",
      exclude: "retweets,replies",
    });
    if (nextToken) params.set("pagination_token", nextToken);

    const pageStart = Date.now();
    let r;
    try {
      r = await fetchWithTimeout(
        `https://api.twitter.com/2/users/${userId}/tweets?${params}`,
        { headers: { Authorization: `Bearer ${bearer}` } },
        PER_X_CALL_TIMEOUT_MS,
      );
    } catch (e) {
      console.log(`[scrape-handle] X page ${i + 1} fetch threw: ${e.message || e}`);
      truncated = true;
      break;
    }
    const pageMs = Date.now() - pageStart;

    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      console.log(`[scrape-handle] X page ${i + 1} status=${r.status} (${pageMs}ms): ${detail.slice(0, 200)}`);
      // Soft-fail: return what we have so the user gets *something*.
      truncated = true;
      break;
    }
    const body = await r.json();
    const batch = body?.data || [];
    out.push(...batch);
    console.log(`[scrape-handle] X page ${i + 1}: ${batch.length} tweets in ${pageMs}ms (cum ${out.length})`);

    nextToken = body?.meta?.next_token;
    if (!nextToken) break;
  }
  return { tweets: out, truncated };
}

function chunkTweets(tweets, size) {
  const chunks = [];
  for (let i = 0; i < tweets.length; i += size) {
    chunks.push(tweets.slice(i, i + size));
  }
  return chunks;
}

async function extractIdeasOneChunk(chunkTweets, handle, anthropic, label) {
  if (!chunkTweets.length) return [];
  const lines = chunkTweets.map((t) => `[${t.id}] ${t.created_at} — ${t.text}`);
  const userMsg = `Extract trading ideas from these ${chunkTweets.length} tweets by @${handle}:\n\n${lines.join("\n\n")}`;
  const system = [
    `You are extracting trading ideas from a Twitter user's recent posts.`,
    `For each tweet that contains a SPECIFIC, ACTIONABLE thesis on a publicly-traded stock (with ticker), emit ONE entry. Skip:`,
    `- generic market commentary with no ticker`,
    `- retweets without added commentary`,
    `- tweets without a directional view`,
    `Sort the output by date descending. Cap at 50 ideas.`,
  ].join("\n");

  const t0 = Date.now();
  try {
    const resp = await Promise.race([
      anthropic.messages.create({
        model: "claude-haiku-4-5",
        max_tokens: 8000,
        system,
        output_config: { format: { type: "json_schema", schema: EXTRACTION_SCHEMA } },
        messages: [{ role: "user", content: userMsg }],
      }),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("anthropic_wall_timeout")), ANTHROPIC_WALL_MS),
      ),
    ]);
    console.log(`[scrape-handle] chunk ${label}: ${Date.now() - t0}ms (in=${resp.usage?.input_tokens} out=${resp.usage?.output_tokens})`);
    const textBlock = resp.content.find((b) => b.type === "text");
    if (!textBlock) return [];
    const parsed = JSON.parse(textBlock.text);
    return Array.isArray(parsed?.ideas) ? parsed.ideas : [];
  } catch (e) {
    console.log(`[scrape-handle] chunk ${label} FAILED after ${Date.now() - t0}ms: ${e.message || e}`);
    return []; // partial degradation — other chunks may still succeed
  }
}

async function extractIdeas(tweets, handle, anthropicKey) {
  const anthropic = new Anthropic({
    apiKey: anthropicKey,
    timeout: ANTHROPIC_TIMEOUT_MS,
    maxRetries: ANTHROPIC_MAX_RETRIES,
  });

  // Sort by date desc first so chunk 1 = newest tweets (most valuable if a chunk fails).
  const sorted = [...tweets].sort((a, b) =>
    (a.created_at || "") < (b.created_at || "") ? 1 : -1,
  );

  const chunks = chunkTweets(sorted, ANTHROPIC_CHUNK_SIZE);
  console.log(`[scrape-handle] extracting ideas in ${chunks.length} parallel chunk(s) of up to ${ANTHROPIC_CHUNK_SIZE}`);

  const results = await Promise.all(
    chunks.map((c, i) => extractIdeasOneChunk(c, handle, anthropic, String(i + 1))),
  );
  const rawIdeas = results.flat();

  // Dedupe by tweet_id (same tweet shouldn't end up in two chunks, but defensive)
  const byId = new Map();
  for (const i of rawIdeas) {
    if (i && i.ticker && i.tweet_id && !byId.has(i.tweet_id)) {
      byId.set(i.tweet_id, i);
    }
  }

  return Array.from(byId.values())
    .map((i) => ({
      tweet_id: i.tweet_id,
      date: i.date,
      ticker: String(i.ticker || "").toUpperCase(),
      stance: String(i.stance || "neutral").toLowerCase(),
      summary: i.summary || "",
      handle,
      tweet_url: `https://x.com/${handle}/status/${i.tweet_id}`,
    }))
    .sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0))
    .slice(0, 50);
}

export default async function handler(req, res) {
  const T0 = Date.now();
  if (req.method !== "GET" && req.method !== "POST") {
    return jsonResponse(res, 405, { error: "Method not allowed" });
  }

  const rawHandle = (req.query?.handle || req.body?.handle || "").toString().trim().replace(/^@/, "");
  if (!rawHandle || !HANDLE_RE.test(rawHandle)) {
    return jsonResponse(res, 400, { error: "Invalid handle. Must match ^[A-Za-z0-9_]{1,15}$" });
  }

  const bearer = process.env.TWITTER_BEARER_TOKEN;
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  if (!bearer || !anthropicKey) {
    return jsonResponse(res, 500, {
      error: "Server not configured",
      detail: "TWITTER_BEARER_TOKEN or ANTHROPIC_API_KEY missing in Vercel env vars.",
    });
  }

  console.log(`[scrape-handle] start handle=${rawHandle}`);

  try {
    const tUserStart = Date.now();
    const user = await lookupUser(rawHandle, bearer);
    console.log(`[scrape-handle] user lookup: ${Date.now() - tUserStart}ms (id=${user.id})`);

    const startIso = new Date(Date.now() - NINETY_DAYS_MS).toISOString();
    const xBudgetDeadline = Date.now() + X_FETCH_BUDGET_MS;
    const { tweets, truncated } = await fetchTimeline(user.id, bearer, startIso, xBudgetDeadline);
    console.log(`[scrape-handle] X fetch total: ${tweets.length} tweets (truncated=${truncated})`);

    if (!tweets.length) {
      return jsonResponse(res, 200, {
        handle: rawHandle,
        user_id: user.id,
        user_name: user.name || null,
        n_tweets: 0,
        n_ideas: 0,
        ideas: [],
        truncated,
        elapsed_ms: Date.now() - T0,
      });
    }

    let ideas = [];
    try {
      ideas = await extractIdeas(tweets, rawHandle, anthropicKey);
    } catch (e) {
      console.log(`[scrape-handle] extractIdeas threw: ${e.message || e}`);
      // Return what we got from X with empty ideas; better than 504.
      return jsonResponse(res, 200, {
        handle: rawHandle,
        user_id: user.id,
        user_name: user.name || null,
        n_tweets: tweets.length,
        n_ideas: 0,
        ideas: [],
        truncated: true,
        warning: "Idea extraction failed within timeout — try again or pick a less prolific handle.",
        elapsed_ms: Date.now() - T0,
      });
    }

    console.log(`[scrape-handle] DONE handle=${rawHandle} total=${Date.now() - T0}ms n_tweets=${tweets.length} n_ideas=${ideas.length}`);
    return jsonResponse(res, 200, {
      handle: rawHandle,
      user_id: user.id,
      user_name: user.name || null,
      n_tweets: tweets.length,
      n_ideas: ideas.length,
      ideas,
      truncated,
      elapsed_ms: Date.now() - T0,
    });
  } catch (e) {
    console.log(`[scrape-handle] FAILED handle=${rawHandle} total=${Date.now() - T0}ms error=${e.message || e}`);
    return jsonResponse(res, e.status || 500, {
      error: e.message || "Scrape failed",
      detail: e.detail || undefined,
    });
  }
}
