# Deployment

Target: **Render free web service** (Docker) + **Neon Postgres** (pgvector), the same as P1.

> Status: **live** at https://policypilot-mqvp.onrender.com (Render free, Singapore; Neon free
> Postgres 17, Singapore). Handbook ingested from a local machine against Neon.

## Steps

1. **Push to GitHub.** From `D:\AI Projects\policypilot`, add a remote and push with tags.
2. **Neon.** Create a project and copy the **direct** connection string (connection pooling
   off).
3. **Migrate and ingest from your machine:**
   ```powershell
   $env:DATABASE_URL="<neon url>"; $env:EMBED_PROVIDER="openai_compat"
   uv run alembic upgrade head
   uv run python -m policypilot.cli ingest
   ```
   Embeddings are computed once at ingest. Only the question is embedded per request.
4. **Render.** New → Web Service → Docker. Set these environment variables:

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | the Neon URL |
   | `ENV` | `prod` |
   | `LLM_PROVIDER` | `openai_compat` |
   | `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
   | `LLM_MODEL` | `gemini-3.1-flash-lite` |
   | `LLM_API_KEY` | your Gemini key |
   | `EMBED_PROVIDER` | `openai_compat` |
   | `JWT_SECRET` | 32+ random bytes |
   | `LOG_SALT` | random |
   | `WORKER_MODE` | `off` |
   | `DEV_TOKENS` | `false` |
   | `DEMO_PERSONAS` | `true` only for a public demo on synthetic data (persona sign-in, never admin) |

5. **Verify.** `/readyz` should return 200. The demo page (`/ui`) needs `DEMO_PERSONAS=true`.

## Production notes

- **Real SSO:** replace HS256 dev tokens with your IdP's RS256 JWTs (OIDC). Verify `iss`, `aud`
  and `exp`, and map the IdP's group claim to `groups`.
  `policypilot/auth.py:verify_token` is the only place that changes.
- **Provider data retention:** questions and retrieved excerpts go to the model provider. Use a
  paid tier with no training on your data before indexing real policies.
