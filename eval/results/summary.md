# StaleGuard eval results

- **Runs combined:** guarded, verify_all
- **Dataset:** 51 items, hand-verified (eval/dataset.jsonl)
- **Judge model(s) that answered, per config:** verify_all: groq/openai/gpt-oss-120b, groq/openai/gpt-oss-20b; guarded: groq/openai/gpt-oss-120b, groq/openai/gpt-oss-20b; guarded_warm: groq/openai/gpt-oss-120b
- **Stale threshold:** STALE claim with correction and confidence ≥ 0.7
- **Tavily:** basic depth, max_results=5, include_published_date=True

## Configurations

- `verify_all`: Baseline: every extracted claim is searched and judged; no volatility filter, no verdict cache.
- `guarded`: StaleGuard: volatility filter (rules + small model), up to N most volatile claims, verdict cache (cold).
- `guarded_warm`: StaleGuard, second pass over the same answers: verdict cache warm (repeated questions).

## Results

| Config | n | Stale recall | Stale precision | False-alarm rate | Median added latency | Tokens / answer | Tavily credits / answer | Tavily credits total | Errors |
|---|---|---|---|---|---|---|---|---|---|
| `verify_all` | 51 | 100% | 93% | 9% | 6,127 ms | 2,877 | 1.57 | 80 | 0 |
| `guarded` | 51 | 93% | 96% | 4% | 4,881 ms | 2,403 | 1.14 | 58 | 0 |
| `guarded_warm` | 51 | 96% | 93% | 9% | 1,143 ms | 1,039 | 0.16 | 8 | 0 |

Confusion counts (TP / FP / FN / TN): `verify_all` 28/2/0/21; `guarded` 26/1/2/22; `guarded_warm` 27/2/1/21

## Misclassified items

| Config | Item | Label | Predicted | Flagged claims |
|---|---|---|---|---|
| `verify_all` | current-go-latest | current | stale | Go 1.27 is the newest Go release. |
| `verify_all` | current-django-lts | current | stale | Django 5.2 is the current long-term support release. |
| `guarded` | role-nestle-ceo | stale | current | - |
| `guarded` | statistic-boe-bank-rate | stale | current | - |
| `guarded` | current-go-latest | current | stale | Go 1.27 is the newest Go release. |
| `guarded_warm` | statistic-boe-bank-rate | stale | current | - |
| `guarded_warm` | history-intel-gelsinger | current | stale | Pat Gelsinger served as Intel's CEO from February 2021. |
| `guarded_warm` | current-go-latest | current | stale | Go 1.27 is the newest Go release. |
