# StaleGuard eval results

- **Run:** 2026-11-20 07:20 UTC
- **Dataset:** 51 items (28 stale / 23 current), hand-verified
- **Small model:** gemini/gemini-3.5-flash-lite
- **Judge model(s) that answered:** groq/openai/gpt-oss-120b, groq/openai/gpt-oss-20b (⚠ Bifrost failover mixed judge models)
- **Stale threshold:** STALE claim with correction and confidence ≥ 0.7
- **Tavily:** basic depth, max_results=5, include_published_date=True
- **Wall time:** 867 s

## Configurations

- `guarded`: StaleGuard: volatility filter (rules + small model), up to N most volatile claims, verdict cache (cold).
- `guarded_warm`: StaleGuard, second pass over the same answers: verdict cache warm (repeated questions).

## Results

| Config | n | Stale recall | Stale precision | False-alarm rate | Median added latency | Tokens / answer | Tavily credits / answer | Tavily credits total | Errors |
|---|---|---|---|---|---|---|---|---|---|
| `guarded` | 51 | 93% | 96% | 4% | 4,881 ms | 2,403 | 1.14 | 58 | 0 |
| `guarded_warm` | 51 | 96% | 93% | 9% | 1,143 ms | 1,039 | 0.16 | 8 | 0 |

Confusion counts (TP / FP / FN / TN): `guarded` 26/1/2/22; `guarded_warm` 27/2/1/21

## Misclassified items

| Config | Item | Label | Predicted | Flagged claims |
|---|---|---|---|---|
| `guarded` | role-nestle-ceo | stale | current | - |
| `guarded` | statistic-boe-bank-rate | stale | current | - |
| `guarded` | current-go-latest | current | stale | Go 1.27 is the newest Go release. |
| `guarded_warm` | statistic-boe-bank-rate | stale | current | - |
| `guarded_warm` | history-intel-gelsinger | current | stale | Pat Gelsinger served as Intel's CEO from February 2021. |
| `guarded_warm` | current-go-latest | current | stale | Go 1.27 is the newest Go release. |
