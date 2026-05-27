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

import Anthropic from "@anthropic-ai/sdk";

export const config = {
  maxDuration: 60,
};

const HANDLE_RE = /^[A-Za-z0-9_]{1,15}$/;
const NINETY_DAYS_MS = 90 * 24 * 60 * 60 * 1000;
const MAX_PAGES = 8; // safety cap → up to 800 tweets

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

async function lookupUser(handle, bearer) {
  const r = await fetch(`https://api.twitter.com/2/users/by/username/${handle}`, {
    headers: { Authorization: `Bearer ${bearer}` },
  });
  if (!r.ok) {
    const detail = await r.text();
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

async function fetchTimeline(userId, bearer, startIso) {
  const out = [];
  let nextToken = null;
  for (let i = 0; i < MAX_PAGES; i++) {
    const params = new URLSearchParams({
      max_results: "100",
      start_time: startIso,
      "tweet.fields": "created_at,public_metrics",
      exclude: "retweets,replies",
    });
    if (nextToken) params.set("pagination_token", nextToken);
    const r = await fetch(`https://api.twitter.com/2/users/${userId}/tweets?${params}`, {
      headers: { Authorization: `Bearer ${bearer}` },
    });
    if (!r.ok) {
      // Soft-fail on pagination errors — return what we have.
      break;
    }
    const body = await r.json();
    const batch = body?.data || [];
    out.push(...batch);
    nextToken = body?.meta?.next_token;
    if (!nextToken) break;
  }
  return out;
}

async function extractIdeas(tweets, handle, anthropicKey) {
  const anthropic = new Anthropic({ apiKey: anthropicKey });

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

  const resp = await anthropic.messages.create({
    model: "claude-haiku-4-5",
    max_tokens: 8000,
    system,
    output_config: {
      format: { type: "json_schema", schema: EXTRACTION_SCHEMA },
    },
    messages: [{ role: "user", content: userMsg }],
  });

  const textBlock = resp.content.find((b) => b.type === "text");
  if (!textBlock) return [];
  let parsed;
  try {
    parsed = JSON.parse(textBlock.text);
  } catch (e) {
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

  try {
    const user = await lookupUser(rawHandle, bearer);
    const startIso = new Date(Date.now() - NINETY_DAYS_MS).toISOString();
    const tweets = await fetchTimeline(user.id, bearer, startIso);

    if (!tweets.length) {
      return jsonResponse(res, 200, {
        handle: rawHandle,
        user_id: user.id,
        user_name: user.name || null,
        n_tweets: 0,
        ideas: [],
      });
    }

    const ideas = await extractIdeas(tweets, rawHandle, anthropicKey);

    return jsonResponse(res, 200, {
      handle: rawHandle,
      user_id: user.id,
      user_name: user.name || null,
      n_tweets: tweets.length,
      n_ideas: ideas.length,
      ideas,
    });
  } catch (e) {
    return jsonResponse(res, e.status || 500, {
      error: e.message || "Scrape failed",
      detail: e.detail || undefined,
    });
  }
}
