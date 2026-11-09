# Bifrost & Tavily — verified behaviour

Checked 2026-11-09 against docs.getbifrost.ai, `github.com/maximhq/bifrost@main` source, and docs.tavily.com.
Items marked **⚠ vs SPEC** contradict or refine SPEC.md. Items marked **TODO-verify-live** must be confirmed against a running instance before code depends on them.

## 1. Running Bifrost

- We pin `maximhq/bifrost:v2.4.0` (v2.5.0 was only 3 days old on 2026-11-09).
- Docker: `docker run -p 8080:8080 -v ./data:/app/data maximhq/bifrost` — `/app/data` is the app dir; `config.json` goes at `/app/data/config.json`. Pin a tag (e.g. `maximhq/bifrost:v2.4.0`) rather than `latest`.
- NPX: `npx -y @maximhq/bifrost -app-dir ./data` (flags: `-port`, `-host`, `-log-level`, `-log-style`). Docker equivalents are env vars `APP_PORT`, `APP_HOST`, `LOG_LEVEL`, `LOG_STYLE`.
- Default app dir without the flag: `%APPDATA%\bifrost` on Windows.
- Web UI at `http://localhost:8080`.
- **Config loading:** if `config_store` is omitted, Bifrost creates a SQLite `config.db` and *seeds* it from `config.json` at startup. UI/API edits persist in the DB, and editing an entity in the file overwrites its DB copy. Set `"config_store": {"enabled": false}` for file-only mode (no UI edits, restart to apply). **Decision for us:** keep the default (DB-backed). Then `config.json` is the reproducible source and the UI still works for the demo. `data/config.db` and `data/logs.db` are gitignored.

### Auth in v2.4.0 — **⚠ found live, not in the quickstart docs**
Confirmed against a running `v2.4.0` and `transports/config.schema.json`:
- **Inference requires a virtual key by default.** `client.enforce_auth_on_inference` defaults to `true` on new deployments, and without a key you get `401 virtual key is required ... x-bf-vk`. We set `"client": {"enforce_auth_on_inference": false}`, which is fine for a local prototype. (Roadmap: a governance virtual key per stage, for per-stage budgets.)
- **`/api/*` (management, incl. cache eviction) is locked** until dashboard auth is enabled. In OSS with no admin account, *every* `/api` call must send `X-Bifrost-Setup-Token: <setup_token>`. We set `"setup_token": "env.BIFROST_SETUP_TOKEN"`. Verified: `GET /api/providers` returns 403 without the header and 200 with it.
- With auth enforcement off, the OpenAI SDK's mandatory `Authorization: Bearer <anything>` is ignored. Verified with a dummy value.
- `/health` is unauthenticated and returns `{"components":{"db_pings":"ok"},"status":"ok"}`. The image ships its own Docker healthcheck.
- Bootstrap takes about 25 s (DB migrations + model catalog), so poll `/health`.
- **Confirmed live (2026-11-09, real key):** a successful chat response carries top-level `extra_fields` (`request_type`, `routing_info{provider,model,key}`, `provider`, `resolved_model_used`, `latency`, `upstream_latency`, `provider_response_headers`). `openai` SDK: `resp.to_dict()["extra_fields"]`.
- **Gemini Flash thinks by default:** a 2-word reply used 138 of 142 completion tokens on reasoning (~2 s). Set `reasoning_effort` low/none for extract/classify in step 3.
- **Free-tier key + `gemini-3.1-pro-preview` → `429`**, quota `limit: 0` for gemini-3.1-pro. Confirms Pro is paid-only.
- Error responses also carry top-level `extra_fields` (`provider`, `original_model_requested`, `resolved_model_used`, `request_type`, `error_type`). That's evidence that success responses use the same top-level key.

### Providers schema (Gemini)
Gemini (AI Studio) is a first-class provider: config key `gemini`, model prefix `gemini/`. It supports chat and embeddings (`/v1beta/models/{model}:embedContent`; `dimensions` maps to `outputDimensionality`).
```json
"providers": {
  "gemini": {
    "keys": [ { "name": "gemini-ai-studio", "value": "env.GEMINI_API_KEY", "models": ["*"], "weight": 1.0 } ]
  }
}
```
- `"value": "env.VAR"` reads the secret from the container env, so keys stay in `.env`.
- An empty env var makes the key invalid. Bifrost logs `no valid keys found for provider: gemini` and chat returns `400 no_key_supports_model`.

### Gemini models (ai.google.dev / aistudio.google.com docs, checked 2026-11-09)
| Use | Model | Free tier |
|---|---|---|
| extract / classify (was) | `gemini-3.6-flash` (stable) | ✅ (20 RPD) |
| judge / rewrite | `gemini-3.1-pro-preview` (preview; only current Pro) | ❌ **paid only**, so the judge moved to Groq (below) |
| **extract / classify (now)** | `gemini-3.5-flash-lite` (stable) | ✅ |
| cache embeddings | `gemini-embedding-001` | ✅ |
- **Free-tier limits hit live (2026-11-09).** They are per model per project, and the docs don't publish them, so these come from 429 bodies:
  - `gemini-3.6-flash`: **5 requests/min, 20 requests/day**. Too few for the judge in dev, the eval or demo rehearsals.
  - `gemini-3.5-flash-lite`: works for extract/classify/judge, and was not throttled across ~20 calls in a few minutes. Exact limits unknown.
  - A 429 for the daily quota is still retried by Bifrost (`max_retries: 3`), so it fails after ~45 s of futile backoff.
- Bifrost `providers.gemini.network_config`: `max_retries: 3`, `retry_backoff_initial: 5000`, `retry_backoff_max: 30000`. Verified: a per-minute 429 on the judge was retried transparently and succeeded after ~60 s. ⚠ Setting `network_config` partially **zeroes the fields you leave out** (`default_request_timeout_in_seconds` became 0), so all timeouts are now set explicitly.
- Embeddings default to **3072 dims** (128–3072 configurable). `gemini-embedding-001` needs manual normalization below 3072, so we use 3072 and avoid the question. `gemini-embedding-2` (multimodal) auto-normalizes.

### Groq provider (judge model), checked 2026-11-09
- Bifrost docs (docs.getbifrost.ai/providers/supported-providers/groq): provider key `groq`, chat + streaming supported, `reasoning_effort` is passed through. `prompt_cache_key`, `verbosity`, `store` and `service_tier` are silently dropped. Embeddings are **not** supported, so the semantic cache stays on Gemini.
- Config: `providers.groq.keys[0].value = "env.GROQ_API_KEY"`, with its own full `network_config` (retries 3, backoff 2000–20000 ms).
- Models live on the key (`GET api.groq.com/openai/v1/models`): `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` (all 131K context), plus small/speech models. Addressed through Bifrost as `groq/openai/gpt-oss-120b`.
- Verified live through Bifrost: `response_format: json_object`, `temperature: 0` and `reasoning_effort: "low"` all work on the three models above. Each returned a correct STALE verdict and correction on a probe in 0.3–1.0 s.
- Rate limits (console.groq.com/docs/rate-limits; table is for the base plan, so confirm on the account's limits page): each of these models has **30 RPM, 1K RPD, 8K TPM, 200K TPD**. Limits are per model.
- ⚠ Tokens, not requests, are the binding limit. A batched judge call over 2 claims with 5 snippets each used ~3.5–3.9K tokens, so 8K TPM is about 2 judge calls/min and 200K TPD about 50 judge calls/day.
- **Judge = `groq/openai/gpt-oss-120b`.** Live smoke run: corrections were "Python 3.14" and "GPT-5 is OpenAI's newest model" with correct citations. Cold verify is 5.9–6.5 s (Flash-Lite judge: 3.2–3.4 s).

### Chat call
`POST /v1/chat/completions`, OpenAI format. Model is `provider/model` (e.g. `gemini/gemini-3.6-flash`). A bare name also resolves via the model catalog, but we always use the prefixed form.

## 2. Semantic cache plugin

```json
"vector_store": { "enabled": true, "type": "redis", "config": { "addr": "redis:6379" } },
"plugins": [{
  "enabled": true,
  "name": "semantic_cache",
  "config": {
    "provider": "gemini",
    "embedding_model": "gemini-embedding-001",
    "dimension": 3072,
    "ttl": "24h",
    "threshold": 0.8,
    "conversation_history_threshold": 3,
    "exclude_system_prompt": false,
    "cache_by_model": true,
    "cache_by_provider": true,
    "vector_store_namespace": "BifrostSemanticCachePlugin",
    "default_cache_key": "",
    "cache_tool_call_responses": false
  }
}]
```
- `vector_store` is a **top-level** key, not inside the plugin config. The plugin uses the global provider keys for embeddings.
- `ttl` defaults to `5m`. **⚠ vs SPEC:** the demo seeds a cache entry and then hits it later, so set something like `"24h"` or the entry expires mid-demo.
- `default_cache_key: ""` means requests **without** `x-bf-cache-key` bypass the cache entirely.
- **Direct-only mode** (exact-match hash, no embeddings): `dimension: 1` and omit `provider`/`embedding_model`. It still needs a vector store. This is the fallback in SPEC §11.
- `conversation_history_threshold: 3` means conversations longer than 3 messages are never cached. Keep that in mind for demo prompts.

### Semantic cache with Gemini embeddings ✅ (source-checked)
`plugins/semanticcache/main.go`: the plugin has no provider allowlist. `Provider` is any `schemas.ModelProvider`, and embeddings go through Bifrost's own embedding executor (so through the `gemini` provider). Semantic search runs only if `provider != ""`, `embedding_model != ""` and `dimension > 1`. `Init` rejects a provider with `dimension == 0`. **TODO-verify-live (step 2)** with a real key.

### Vector store — **⚠ vs SPEC (resolved: Redis adopted, SPEC updated)**
Supported: Redis/Valkey (needs RediSearch, i.e. `redis/redis-stack`), Weaviate, Qdrant, Pinecone.
- SPEC picks Weaviate. Bifrost's docs recommend Redis for direct mode, and its Redis config is a single `addr` field. Weaviate needs `scheme` + `host` plus gRPC config (port 50051), and its docs example clashes with Bifrost on port 8080.
- **Recommendation:** use `redis/redis-stack` instead of Weaviate. It's a smaller container, has fewer moving parts, and serves both modes. It is a swap of one container behind the same Bifrost config key, not an architecture change. Needs the user's OK (SPEC §4 names Weaviate).
- Weaviate config, if kept: `{"type":"weaviate","config":{"scheme":"http","host":"weaviate:8080","grpc_config":{"host":"weaviate:50051","secured":false}}}`.

### Request headers ✅ (matches SPEC)
| Header | Effect |
|---|---|
| `x-bf-cache-key` | Cache partition. Required unless `default_cache_key` is set. |
| `x-bf-cache-ttl` | Per-request TTL (duration or seconds). |
| `x-bf-cache-threshold` | Can only *raise* the similarity threshold. |
| `x-bf-cache-no-store` | `true` = read from cache but don't write. |
| `x-bf-cache-type` | `direct` or `semantic` only. |

### `extra_fields.cache_debug` ✅ (matches SPEC, with more fields)
Go struct `BifrostCacheMetadata` (`BifrostCacheDebug` is a deprecated alias), tagged `json:"cache_debug,omitempty"` inside `BifrostResponseExtraFields`:
```
cache_hit: bool            (always)
cache_id: str?             (docs: always present)
hit_type: "direct"|"semantic"?   (on hit)
threshold, similarity: float?    (semantic hit)
requested_provider, requested_model, provider_used, model_used: str?
input_tokens: int?
cache_hit_latency: int?
```
- The same data is also exposed as `x-bifrost-cache-*` **response headers**. That's useful if the JSON field ever gets stripped.
- Streaming: only the final chunk carries it.
- ✅ **Verified live (step 2, 2026-11-09)** with `gemini-embedding-001`; recorded in `tests/fixtures/bifrost_chat_*.json`:
  - miss: `{"cache_hit": false, "cache_id": "<uuid>"}`, so **`cache_id` is present on a miss** (the entry just written).
  - exact repeat: `hit_type: "direct"`, same `cache_id`, `cache_hit_latency` ~1–3 ms.
  - paraphrase ("Which city is Australia's capital?"): `hit_type: "semantic"`, `similarity: 0.96`, `model_used: gemini-embedding-001`, same `cache_id`, ~530 ms (embedding call).
  - No `x-bf-cache-key` means no `cache_debug` at all (cache bypassed).
  - **`cache_id` is deterministic**: after eviction, re-asking recreates the *same* id. So the id identifies the request, not the stored answer.

### Eviction ✅ (matches SPEC)
Confirmed in `transports/bifrost-http/handlers/cache.go`:
- `DELETE /api/cache/clear/{cacheId}` removes one entry. Returns 400 if the plugin isn't loaded or the ID is missing, and 500 on failure.
- `DELETE /api/cache/clear-by-key/{cacheKey}` removes all entries under a cache key (URL-encode it). Useful as a coarser fallback.
- ✅ **Verified live (step 2):** without `X-Bifrost-Setup-Token` you get `401`. With it you get `200 {"message":"Cache cleared successfully"}`, and the next identical request is a **miss**. ⚠ It also returns 200 for already-evicted or nonexistent ids, so a 200 is not proof anything was removed. To confirm an eviction, re-query with `x-bf-cache-no-store: true` and check for a miss.
- Disabling the plugin via `PUT /api/plugins/semantic_cache` doesn't clear entries.

## 3. Tavily (`tavily-python`)

`TavilyClient` / `AsyncTavilyClient(api_key).search(query, ...)`. Relevant params:
| Param | Values / default | Notes |
|---|---|---|
| `search_depth` | `basic` (default), `advanced`, `fast`, `ultra-fast` | Only `advanced` costs 2 credits; the others cost 1. |
| `topic` | `general` (default), `news`, `finance` | |
| `time_range` | `day/week/month/year` or `d/w/m/y` | |
| `start_date`, `end_date` | `YYYY-MM-DD` | |
| `max_results` | default **10**, 0–20 | SPEC wants 5, so pass it explicitly. |
| `include_answer` | `False` / `True`/`"basic"` / `"advanced"` | |
| `include_domains` / `exclude_domains` | lists (max 300 / 150) | |
| `include_domains_mode` | `restrict` (default) / `prefer` | `prefer` fits a trusted-domain *preference* better than a hard filter. |
| `chunks_per_source`, `country`, `language`, `max_age_hours`, `include_raw_content`, `include_usage`, `timeout` | | |

- **⚠ vs SPEC:** there is **no `days` param** (SPEC doesn't use it, but older examples online do).
- **⚠ vs SPEC:** `published_date` is returned only when `include_published_date=True`. It's turned on automatically for `topic="news"`. For `general` topic we must pass `include_published_date=True` explicitly, or evidence won't carry dates. ✅ **Verified live (tavily-python 0.8.5, 2026-11-09):** `include_published_date` is not a named parameter, but `search()` forwards `**kwargs` to the API, and every result came back with `published_date`. Dates are **RFC 2822** (`Fri, 04 Jul 2025 20:09:07 GMT`), so `verify.normalize_date` converts them to ISO. 0.8.5 still accepts `days` (legacy); we never pass it. Results also carry `id` and `raw_content` keys. Recorded responses are in `tests/fixtures/tavily/`.
- Response: `{query, answer?, results[{title, url, content, score, published_date?, raw_content?}], response_time, request_id}`.
