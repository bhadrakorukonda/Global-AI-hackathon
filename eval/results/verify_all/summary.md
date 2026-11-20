# StaleGuard eval results

- **Run:** 2026-11-20 21:18 UTC
- **Dataset:** 51 items (28 stale / 23 current), hand-verified
- **Small model:** gemini/gemini-3.5-flash-lite
- **Judge model(s) that answered:** groq/openai/gpt-oss-120b, groq/openai/gpt-oss-20b (⚠ Bifrost failover mixed judge models)
- **Stale threshold:** STALE claim with correction and confidence ≥ 0.7
- **Tavily:** basic depth, max_results=5, include_published_date=True
- **Wall time:** 659 s

## Configurations

- `verify_all`: Baseline: every extracted claim is searched and judged; no volatility filter, no verdict cache.

## Results

| Config | n | Stale recall | Stale precision | False-alarm rate | Median added latency | Tokens / answer | Tavily credits / answer | Tavily credits total | Errors |
|---|---|---|---|---|---|---|---|---|---|
| `verify_all` | 51 | 100% | 93% | 9% | 6,127 ms | 2,877 | 1.57 | 80 | 0 |

Confusion counts (TP / FP / FN / TN): `verify_all` 28/2/0/21

## Misclassified items

| Config | Item | Label | Predicted | Flagged claims |
|---|---|---|---|---|
| `verify_all` | current-go-latest | current | stale | Go 1.27 is the newest Go release. |
| `verify_all` | current-django-lts | current | stale | Django 5.2 is the current long-term support release. |
