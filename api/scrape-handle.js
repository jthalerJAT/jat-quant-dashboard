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
const PER_X_CALL_TIMEOUT_MS = 8000;   // 8s per X HTTP request
const X_FETCH_BUDGET_MS = 22000;      // total wall-clock cap for all X calls
const ANTHROPIC_TIMEOUT_MS = 30000;   // 30s on the batched extraction call

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

async function extractIdeas(tweets, handle, anthropicKey) {
  const anthropic = new Anthropic({ apiKey: anthropicKey, timeout: ANTHROPIC_TIMEOUT_MS });

  const lines = tweets.map((t) => `[${t.id}] ${t.created_at} — ${t.text}`);
  const userMsg = `Extract trading ideas from these ${tweets.length} tweets by @${handle} (most recent ~90 days):\n\n${lines.join("\n\n")}`;

  const system = [
    `You are extracting trading ideas from a Twitter user's recent posts.`,
    `For each tweet that contains a SPECIFIC, ACTIONABLE thesis on a publicly-traded stock (with ticker), emit ONE entry. Skip:`,
    `- generic market commentary with no ticker`,
    `- retweets without added commentary`,
    `- tweets without a directional view`,
    `Sort the output by date descending. Cap at 50 ideas.`,
  ].join("\n");

  const t0 = Date.now();
  const resp = await anthropic.messages.create({
    model: "claude-haiku-4-5",
    max_tokens: 8000,
    system,
    output_config: {
      format: { type: "json_schema", schema: EXTRACTION_SCHEMA },
    },
    messages: [{ role: "user", content: userMsg }],
  });
  console.log(`[scrape-handle] Anthropic call: ${Date.now() - t0}ms (input ${resp.usage?.input_tokens}, output ${resp.usage?.output_tokens})`);

  const textBlock = resp.content.find((b) => b.type === "text");
  if (!textBlock) return [];
  let parsed;
  try {
    parsed = JSON.parse(textBlock.text);
  } catch (e) {
    console.log(`[scrape-handle] JSON parse failed: ${e.message}`);
    return [];
  }
  const ideas = Array.isArray(parsed?.ideas) ? parsed.ideas : [];

  return ideas
    .filter((i) => i && i.ticker && i.tweet_id)
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
