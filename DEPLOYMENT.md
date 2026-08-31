# Deploying Analyst Copilot

Generic instructions for running this on any Docker-friendly host. A
`Dockerfile` and `.dockerignore` are at the repo root.

## What the container needs

**Environment variables** (set as secrets/env vars in your host's
dashboard -- never commit them):

- `LLM_PROVIDER` -- optional, defaults to `fireworks`. One of `fireworks`,
  `bedrock`, `bedrock_openai`, `azure` -- see `backend/llm.py`'s top-of-file
  provider config for exactly which env vars each one needs
  (`FIREWORKS_API_KEY`; `AWS_REGION` + `AWS_BEARER_TOKEN_BEDROCK` for either
  Bedrock variant; `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` +
  `AZURE_OPENAI_DEPLOYMENT` for Azure).
- `EMBEDDING_MODEL` -- optional, defaults to `normal` (BGE-small, baked
  into the image at build time -- see the Dockerfile).
- `DATABASE_URL` -- optional, a Postgres connection string. See
  "Persistent storage" below -- lets the app survive without a mounted
  volume.

**Persistent storage.** `data/uploads/` (raw uploaded filings) and
`data/indexes/` (their BM25/FAISS indexes) are created at runtime and are
NOT baked into the image -- without a persistent volume mounted at
`/app/data`, every filing you index disappears on the next deploy or
container restart and has to be re-uploaded. Mount a volume there on
whichever host you use.

**Or skip the volume and use Postgres instead.** Set `DATABASE_URL` (a
normal Postgres connection string, `postgresql://user:pass@host:5432/db`)
and the app additively syncs every indexed filing there too (see
`backend/postgres_store.py`); on startup, any filing Postgres knows about
that isn't on local disk gets pulled back down automatically (see
`hydrate_from_postgres` in `backend/main.py`'s startup handler). This is
what makes a host WITHOUT persistent volumes (or a fresh container after
every deploy) still come up with all your indexed filings. A volume and
Postgres both being set is fine too -- local disk stays a warm cache
either way, Postgres is the durable copy. Postgres support requires the
`vector` extension (pgvector) and only stores 384-dim embeddings, i.e.
filings indexed with `EMBEDDING_MODEL=normal` (the default) -- see the
docstring at the top of `backend/postgres_store.py` for why.

To backfill filings you already indexed locally into a freshly provisioned
Postgres database:
```bash
DATABASE_URL=postgresql://... python scripts/backfill_postgres.py
```
Safe to re-run; it upserts, it doesn't duplicate.

### Azure specifically

The $200 free-tier credit covers this comfortably for a hackathon-length
deployment:
1. **Azure Database for PostgreSQL - Flexible Server** (Burstable B1ms
   tier is enough here) -- create it, then connect and run
   `CREATE EXTENSION vector;` once (or let `postgres_store.ensure_schema()`
   do it automatically on first app startup -- it runs
   `CREATE EXTENSION IF NOT EXISTS vector;` itself). Copy the connection
   string into `DATABASE_URL`.
2. **Azure Container Apps** (or Web App for Containers) -- point it at
   this repo's `Dockerfile`, either via a container registry push or
   Azure's "build from GitHub repo" flow. Set `DATABASE_URL` and your LLM
   provider's env vars as secrets in the Container App's configuration.
3. Run the backfill script once (from your own machine, pointed at the
   Azure Postgres connection string, or as a one-off Container Apps job)
   to push your locally-indexed filings up before the first deploy.

**Port.** The container reads `$PORT` and binds to it (falls back to
`8000` if unset). Most PaaS hosts inject `PORT` automatically; for a bare
`docker run` you control it yourself with `-p`.

**Health check.** `GET /api/health` returns `{"status": "ok", ...}` --
point your host's health check at this path.

## Build and run locally (sanity check before deploying)

```bash
docker build -t analyst-copilot .
docker run -p 8000:8000 \
  -e FIREWORKS_API_KEY=your-key-here \
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
  `fly.toml`. Set secrets with `fly secrets set FIREWORKS_API_KEY=...`.
- **Railway**: New Project -> Deploy from GitHub repo -> it detects the
  Dockerfile automatically. Add a Volume from the service's Settings,
  mount path `/app/data`. Set env vars under Variables.
