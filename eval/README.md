# StaleGuard eval

Measures whether StaleGuard catches stale answers without crying wolf, and what the volatility filter and
verdict cache save compared with checking every claim (SPEC §8).

## Dataset: `dataset.jsonl`

One JSON object per line. Lines starting with `//` are comments.

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique, readable slug, e.g. `price-gpt4o-input`. |
| `question` | yes | What a user asked. |
| `answer_text` | yes | The answer under test, written the way an assistant would answer. |
| `label` | yes | `stale`: the answer contains a claim that is no longer true today. `current`: everything in it is still true. |
| `class` | yes | Class of the fact the label hinges on: `version`, `role`, `price`, `statistic`, `policy`, `event_status`, `relative_time`, `static`. |
| `ground_truth` | yes | The true fact as of `checked_on`, in one sentence. |
| `source_url` | yes | Authoritative page that shows `ground_truth`. Prefer official sources: vendor pricing pages, release pages, company leadership pages, regulators. |
| `checked_on` | when verified | `YYYY-MM-DD` you checked the source. |
| `verified` | yes | `true` only after **you** opened `source_url` and confirmed `ground_truth`. The runner ignores unverified rows. |
| `notes` | no | Anything useful (why it's tricky, what changed and when). |

### What to put in it (~50 rows)
- **~25 `stale`.** Real facts that changed in roughly the last 18 months. Write the answer as it would have been correct *before* the change. Spread across classes: library and runtime versions, model names and pricing, CEO and leadership changes, regulation milestones, product launches, statistics.
- **~25 `current`.** Correct, up-to-date answers. Include:
  - static facts (should take the fast path, 0 Tavily calls);
  - historical facts that mention old versions or dates but are still true ("Python 3.12 was released in October 2023"). These catch false alarms;
  - current versions, prices and roles stated correctly.
- One fact per row is easiest to label. A row with several claims is fine; the label is `stale` if any of them is.

### Rules
- **Don't let an LLM write the ground truth or the label.** You can use one to draft answer texts, but every `ground_truth` must come from `source_url`, checked by you.
- Re-check fast-moving rows (model names, prices) on the day you run the eval, and update `checked_on`.

## Running

```powershell
docker compose up -d                                              # Bifrost + Redis
uv run python eval/run_eval.py                                    # plan + cost estimate, spends nothing
uv run python eval/run_eval.py --limit 4 --include-unverified --yes   # smoke run on the examples
uv run python eval/run_eval.py --yes                              # full run
```

Configurations (all run in one go by default):

| Config | What it does |
|---|---|
| `verify_all` | Baseline: every extracted claim is searched and judged; no volatility filter, no verdict cache. |
| `guarded` | StaleGuard as shipped: rules + small model filter out static claims, check up to 5 most volatile, verdict cache starts empty. |
| `guarded_warm` | Same answers again with the cache now warm, i.e. the cost of repeated questions. |

Outputs, committed for the README:
- `results/results.jsonl` holds one row per item per config, with the prediction, flagged claims, corrections, latency, tokens, Tavily calls and the models that answered.
- `results/summary.md` holds recall, precision, false-alarm rate, median added latency, tokens and Tavily credits per answer, plus every misclassified item.

### Scoring
An answer counts as predicted `stale` when it has a STALE claim with a correction and judge confidence ≥
`STALE_CONFIDENCE_THRESHOLD` (0.7). That is the same rule the proxy uses before rewriting or blocking.

### Budget (free tiers)
- A full run of 50 items was estimated at about 300 Tavily credits; the real 51-item run used 146 (verify_all 80, guarded 58, guarded_warm 8), because answers average about 1.5 claims. Basic search = 1 credit.
- After separate per-config runs, `uv run python eval/combine_results.py` merges `results/*/results.jsonl` into `results/summary.md`, `results/results.jsonl` and the chart (`results/chart-light.svg`, `results/chart-dark.svg`).
- Groq's free tier allows 8K tokens/min and 200K/day per model. When the judge model runs out, Bifrost fails over to the next model, and `summary.md` says so ("Judge model(s) that answered"). For a clean comparison, all configs should be judged by the same model: run on a fresh quota day, or run one config per day with `--configs`.
