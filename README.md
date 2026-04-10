# JAT Quant Dashboard

A static web dashboard for the **JAT Quant Signal System**. It displays regime status, composite risk-on / risk-off scores, confirmed signals, and pre-threshold “watching” long/short opportunities.

## Data

The UI loads `data/signals_latest.json` (same shape as the signal run audit JSON). Your signal pipeline should overwrite that file each hour so the team always sees the latest run.

## Deploy

This repo is set up for **Vercel** (see `vercel.json`). The `/data/*` path is served with `Cache-Control: no-cache` so JSON updates are not cached at the edge.

## Local preview

Serve the folder with any static server (so `fetch` works), for example:

```bash
npx serve .
```

Then open the URL shown in the terminal.
