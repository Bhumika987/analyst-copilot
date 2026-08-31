# Deploying Analyst Copilot

Generic instructions for running this on any Docker-friendly host. A
`Dockerfile` and `.dockerignore` are at the repo root.

## What the container needs

**Environment variables** (set as secrets/env vars in your host's
dashboard -- never commit them):

- `GROQ_API_KEY` -- required unless you're using a different `LLM_PROVIDER`.
- `LLM_PROVIDER` -- optional, defaults to `groq`. Set to `claude` (needs
  `ANTHROPIC_API_KEY`, defaults to `claude-haiku-4-5`) or one of the other
  providers wired into `backend/llm.py` if you're using those instead.
- `EMBEDDING_MODEL` -- optional, defaults to `normal` (BGE-small, baked
  into the image at build time -- see the Dockerfile).

**Persistent storage.** `data/uploads/` (raw uploaded filings) and
`data/indexes/` (their BM25/FAISS indexes) are created at runtime and are
NOT baked into the image -- without a persistent volume mounted at
`/app/data`, every filing you index disappears on the next deploy or
container restart and has to be re-uploaded. Mount a volume there on
whichever host you use.

**Port.** The container reads `$PORT` and binds to it (falls back to
`8000` if unset). Most PaaS hosts inject `PORT` automatically; for a bare
`docker run` you control it yourself with `-p`.

**Health check.** `GET /api/health` returns `{"status": "ok", ...}` --
point your host's health check at this path.

## Build and run locally (sanity check before deploying)

```bash
docker build -t analyst-copilot .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your-key-here \
  -v analyst-copilot-data:/app/data \
  analyst-copilot
```

Open http://localhost:8000 -- this should behave identically to `python
backend/main.py`, just containerized. Confirm `/api/health` responds before
pushing anywhere.

## Deploying (generic steps, any host)

1. **Push the image to a registry** the host can pull from (Docker Hub,
   GHCR, or the host's own registry):
   ```bash
   docker build -t <registry>/analyst-copilot:latest .
   docker push <registry>/analyst-copilot:latest
   ```
   Most hosts (Render, Railway, Fly.io) can instead build directly from
   your GitHub repo using the root `Dockerfile` -- skip the manual
   push if the host offers that; point it at this repo and it'll pick the
   Dockerfile up automatically.
2. **Create the service** on your host, pointing at the image (or repo).
3. **Set the environment variables** listed above in the host's
   dashboard/secrets manager.
4. **Attach a persistent volume** mounted at `/app/data`.
5. **Set the health check path** to `/api/health` if the host asks.
6. Deploy, then open the assigned URL and try uploading a small filing to
   confirm indexing + a chat question both work end-to-end before relying
   on it.

## Known limitation: no auth

There's currently no login or per-request cost limiting on this app -- the
upload and chat endpoints are open to whoever has the URL, and chat calls
hit a metered LLM API. You chose to leave this open for now; if that
changes, the places to add a gate are `backend/main.py`'s
`/api/filings/upload*` and `/api/chat*` endpoints (a shared-passphrase
header check is the smallest addition that would close this).

## Quick reference: common hosts

These all support "point at a GitHub repo with a Dockerfile" deploys and
persistent volumes -- pick whichever you already have an account on, the
steps above apply to any of them:

- **Render**: New -> Web Service -> connect repo -> it detects the
  Dockerfile automatically. Add a Disk under the service's Disks tab,
  mount path `/app/data`. Set env vars under Environment.
- **Fly.io**: `fly launch` in the repo root detects the Dockerfile and
  scaffolds a `fly.toml`. Add a volume with
  `fly volumes create data --size 1` and mount it at `/app/data` in
  `fly.toml`. Set secrets with `fly secrets set GROQ_API_KEY=...`.
- **Railway**: New Project -> Deploy from GitHub repo -> it detects the
  Dockerfile automatically. Add a Volume from the service's Settings,
  mount path `/app/data`. Set env vars under Variables.
