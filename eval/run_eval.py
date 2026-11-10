"""Run the StaleGuard eval (SPEC §8): verify-every-claim baseline vs volatility-filtered + verdict cache.

Spends real Tavily credits and LLM tokens, so it only runs with --yes; without it, it prints the plan and
a cost estimate. Needs Bifrost up (docker compose up -d) and the keys in .env. Items run one at a time
(free-tier rate limits); Bifrost retries and falls back across providers on its own.

Usage:
  uv run python eval/run_eval.py                       # plan + estimate only
  uv run python eval/run_eval.py --limit 4 --yes       # smoke run on 4 items
  uv run python eval/run_eval.py --yes                 # full run -> eval/results/{results.jsonl,summary.md}

Only rows with "verified": true are used unless --include-unverified is passed.
"""

import argparse
import asyncio
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from staleguard.config import get_settings
from staleguard.evaluation import (
    DatasetError,
    EvalItem,
    ItemResult,
    compute_metrics,
    item_result,
    load_dataset,
    render_summary,
)
from staleguard.factory import build_pipeline
from staleguard.pipeline import Pipeline
from staleguard.schemas import VerifyRequest
from staleguard.verdict_cache import VerdictCache

ROOT = Path(__file__).resolve().parent
CONFIGS = ("verify_all", "guarded", "guarded_warm")
VERIFY_ALL_MAX_CLAIMS = 20  # "every claim", with a sanity cap


async def run_config(name: str, pipeline: Pipeline, items: list[EvalItem], threshold: float, pause: float) -> list[ItemResult]:
    results = []
    for n, item in enumerate(items, start=1):
        try:
            report = await pipeline.run(VerifyRequest(text=item.answer_text, question=item.question))
            r = item_result(item, name, report, threshold)
        except Exception as exc:  # one bad item must not sink the run; it is reported as an error
            r = ItemResult(item_id=item.id, config=name, label=item.label, claim_class=item.claim_class,
                           error=f"{type(exc).__name__}: {exc}"[:300])
        mark = "ERR " if r.error else ("ok  " if r.predicted == r.label else "MISS")
        print(f"  [{name} {n}/{len(items)}] {mark} {item.id:24} label={item.label:7} pred={r.predicted} "
              f"tavily={r.tavily_calls} {r.latency_ms} ms {r.error or ''}", flush=True)
        results.append(r)
        if pause:
            await asyncio.sleep(pause)
    return results


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(ROOT / "dataset.jsonl"))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--configs", default=",".join(CONFIGS), help=f"comma-separated subset of {', '.join(CONFIGS)}")
    p.add_argument("--limit", type=int, default=None, help="only the first N items")
    p.add_argument("--include-unverified", action="store_true", help="also use rows with verified=false (smoke tests only)")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between items (rate limits)")
    p.add_argument("--yes", action="store_true", help="actually run (spends Tavily credits and LLM tokens)")
    args = p.parse_args()

    settings = get_settings()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if unknown := set(configs) - set(CONFIGS):
        p.error(f"unknown config(s): {', '.join(sorted(unknown))}")
    if "guarded_warm" in configs and "guarded" not in configs:
        p.error("guarded_warm reuses the cache filled by guarded; include both")

    try:
        items = load_dataset(args.dataset, include_unverified=args.include_unverified)
    except DatasetError as exc:
        print(f"Dataset has invalid rows:\n{exc}")
        return 2
    items = items[: args.limit] if args.limit else items
    if not items:
        print("No usable items. Rows need \"verified\": true (or pass --include-unverified for a smoke run).")
        return 2

    n_stale = sum(i.label == "stale" for i in items)
    cold = [c for c in configs if c != "guarded_warm"]
    print(f"Dataset: {len(items)} items ({n_stale} stale, {len(items) - n_stale} current); configs: {', '.join(configs)}")
    print(f"Models: small={settings.small_model} judge={settings.judge_model} (fallbacks {settings.judge_model_fallbacks})")
    print(f"Estimate: ~{len(items) * 3 * len(cold)} Tavily credits (≈3 claims/answer; caps: guarded "
          f"{settings.max_claims_per_answer}, verify_all {VERIFY_ALL_MAX_CLAIMS} per answer); "
          f"~{len(items) * len(cold)} judge calls. guarded_warm should cost ~0 Tavily.")
    if not args.yes:
        print("\nPlan only. Re-run with --yes to spend credits.")
        return 0

    results: list[ItemResult] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        if "verify_all" in configs:
            pipeline = build_pipeline(settings, verdict_cache=None, volatility_filter=False, max_claims=VERIFY_ALL_MAX_CLAIMS)
            if pipeline is None:
                print("TAVILY_API_KEY is not set.")
                return 2
            results += await run_config("verify_all", pipeline, items, settings.stale_confidence_threshold, args.pause)
        if "guarded" in configs:
            cache = VerdictCache(Path(tmp) / "eval-verdicts.sqlite3")  # fresh: dev runs must not leak into results
            pipeline = build_pipeline(settings, verdict_cache=cache)
            if pipeline is None:
                print("TAVILY_API_KEY is not set.")
                return 2
            results += await run_config("guarded", pipeline, items, settings.stale_confidence_threshold, args.pause)
            if "guarded_warm" in configs:
                results += await run_config("guarded_warm", pipeline, items, settings.stale_confidence_threshold, args.pause)
            cache.close()

    judge_models = sorted({r.models["judge"] for r in results if "judge" in r.models})
    meta = {
        "Run": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "Dataset": f"{len(items)} items ({n_stale} stale / {len(items) - n_stale} current), "
        + ("hand-verified" if all(i.verified for i in items) else "⚠ includes UNVERIFIED rows: smoke run, not a result"),
        "Small model": settings.small_model,
        "Judge model(s) that answered": ", ".join(judge_models) or "n/a",
        "Stale threshold": f"STALE claim with correction and confidence ≥ {settings.stale_confidence_threshold}",
        "Tavily": "basic depth, max_results=5, include_published_date=True",
        "Wall time": f"{time.perf_counter() - started:.0f} s",
    }
    if len(judge_models) > 1:
        meta["Judge model(s) that answered"] += " (⚠ Bifrost failover mixed judge models)"
    metrics = [compute_metrics(c, results) for c in configs]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
    (out / "summary.md").write_text(render_summary(metrics, results, meta), encoding="utf-8")
    print(f"\nWrote {out / 'results.jsonl'} and {out / 'summary.md'}\n")
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
