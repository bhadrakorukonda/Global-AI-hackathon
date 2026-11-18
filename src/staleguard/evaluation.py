"""Evaluation (SPEC §8): dataset model, answer-level scoring, metrics and the markdown summary.

No network here; eval/run_eval.py drives the real pipeline and calls into this module.

Scoring is per answer: an answer is predicted `stale` when it has at least one STALE claim with a correction
at or above the confidence threshold, the same rule the proxy uses before it rewrites or blocks
(pipeline.act.actionable_stale). So the eval measures what users actually experience.
"""

import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from staleguard.pipeline.act import actionable_stale
from staleguard.schemas import ClaimClass, Report

__all__ = [
    "ConfigMetrics",
    "DatasetError",
    "EvalItem",
    "ItemResult",
    "RowCheck",
    "apply_checks",
    "compute_metrics",
    "item_result",
    "load_dataset",
    "render_summary",
]

Label = Literal["stale", "current"]


class EvalItem(BaseModel):
    """One row of eval/dataset.jsonl. Rows are written and hand-checked by a person, never generated."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer_text: str = Field(min_length=1, description="The answer under test, written as an assistant would.")
    label: Label = Field(description="stale = the answer contains a claim that is no longer true today.")
    claim_class: ClaimClass = Field(alias="class", description="Class of the fact the label hinges on.")
    ground_truth: str = Field(min_length=1, description="The fact as of checked_on, in one sentence.")
    source_url: str = Field(pattern=r"^https?://", description="Authoritative source for ground_truth.")
    checked_on: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date a person checked it.")
    verified: bool = Field(default=False, description="Set true only after a person checked ground_truth at source_url.")
    notes: str = ""


class DatasetError(ValueError):
    """The dataset file has invalid rows (message lists every problem with its line number)."""


def load_dataset(path: str | Path, include_unverified: bool = False) -> list[EvalItem]:
    """Parse and validate the JSONL dataset. Blank lines and lines starting with // are ignored."""
    items: list[EvalItem] = []
    problems: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            item = EvalItem.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            problems.append(f"line {n}: {exc}")
            continue
        if item.id in seen:
            problems.append(f"line {n}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if problems:
        raise DatasetError("\n".join(problems))
    return items if include_unverified else [i for i in items if i.verified]


class RowCheck(BaseModel):
    """One person's check of one dataset row, as exported from the team's Row Checker page."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["checked", "dropped"]
    checked_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    ground_truth: str | None = None
    label: Label | None = None
    note: str | None = None


def apply_checks(items: list[EvalItem], checks: dict[str, RowCheck]) -> tuple[list[EvalItem], list[str]]:
    """Merge human checks into dataset rows. Returns (updated items, row ids that got no check).

    checked -> verified=true with the checker's date, and their ground_truth/label when they changed it.
    dropped -> verified=false with a note, so the runner skips it but the row stays visible for review.
    """
    by_id = {i.id: i for i in items}
    unknown = sorted(set(checks) - set(by_id))
    if unknown:
        raise DatasetError(f"checks for ids not in the dataset: {', '.join(unknown)}")
    out: list[EvalItem] = []
    for item in items:
        check = checks.get(item.id)
        if check is None:
            out.append(item)
            continue
        notes = " | ".join(p for p in (item.notes, check.note and f"checker: {check.note}") if p)
        update: dict[str, object] = {"checked_on": check.checked_on, "notes": notes}
        if check.status == "checked":
            update["verified"] = True
            if check.ground_truth and check.ground_truth.strip():
                update["ground_truth"] = check.ground_truth.strip()
            if check.label:
                update["label"] = check.label
        else:
            update["verified"] = False
            update["notes"] = " | ".join(p for p in (notes, "dropped by checker") if p)
        out.append(EvalItem.model_validate({**item.model_dump(by_alias=True), **update}))
    return out, [i.id for i in items if i.id not in checks]


class ItemResult(BaseModel):
    """Outcome of one dataset item under one pipeline configuration."""

    item_id: str
    config: str
    label: Label
    claim_class: ClaimClass
    predicted: Label | None = None  # None = the run errored
    verdict_summary: str | None = None
    flagged_claims: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    claims_extracted: int = 0
    claims_checked: int = 0
    latency_ms: int = 0
    tokens: int = 0
    tavily_calls: int = 0
    verdict_cache_hits: int = 0
    models: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


def item_result(item: EvalItem, config: str, report: Report, threshold: float) -> ItemResult:
    stale = actionable_stale(report, threshold)
    return ItemResult(
        item_id=item.id,
        config=config,
        label=item.label,
        claim_class=item.claim_class,
        predicted="stale" if stale else "current",
        verdict_summary=report.verdict_summary,
        flagged_claims=[v.claim.text for v in stale],
        corrections=[v.corrected_claim or "" for v in stale],
        claims_extracted=report.claims_extracted,
        claims_checked=report.claims_checked,
        latency_ms=report.latency_ms,
        tokens=sum(report.tokens_by_stage.values()),
        tavily_calls=report.tavily_calls,
        verdict_cache_hits=report.verdict_cache_hits,
        models=report.models_by_stage,
    )


class ConfigMetrics(BaseModel):
    config: str
    n: int
    errors: int
    tp: int
    fp: int
    fn: int
    tn: int
    recall: float | None = Field(description="Stale answers caught / all stale answers.")
    precision: float | None = Field(description="Correct stale flags / all stale flags.")
    false_alarm_rate: float | None = Field(description="Current answers flagged stale / all current answers.")
    median_latency_ms: float | None
    tokens_per_answer: float | None
    tavily_calls_per_answer: float | None
    tavily_credits: int = Field(description="basic search = 1 credit per call.")


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(config: str, results: Iterable[ItemResult]) -> ConfigMetrics:
    rows = [r for r in results if r.config == config]
    ok = [r for r in rows if r.predicted is not None]
    tp = sum(r.label == "stale" and r.predicted == "stale" for r in ok)
    fp = sum(r.label == "current" and r.predicted == "stale" for r in ok)
    fn = sum(r.label == "stale" and r.predicted == "current" for r in ok)
    tn = sum(r.label == "current" and r.predicted == "current" for r in ok)
    calls = sum(r.tavily_calls for r in ok)
    return ConfigMetrics(
        config=config,
        n=len(rows),
        errors=len(rows) - len(ok),
        tp=tp, fp=fp, fn=fn, tn=tn,
        recall=_ratio(tp, tp + fn),
        precision=_ratio(tp, tp + fp),
        false_alarm_rate=_ratio(fp, fp + tn),
        median_latency_ms=statistics.median(r.latency_ms for r in ok) if ok else None,
        tokens_per_answer=_ratio(sum(r.tokens for r in ok), len(ok)),
        tavily_calls_per_answer=_ratio(calls, len(ok)),
        tavily_credits=calls,
    )


def _fmt(v: float | None, pct: bool = False, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.0f}%" if pct else f"{v:,.{digits}f}"


CONFIG_DESCRIPTIONS = {
    "verify_all": "Baseline: every extracted claim is searched and judged; no volatility filter, no verdict cache.",
    "guarded": "StaleGuard: volatility filter (rules + small model), up to N most volatile claims, verdict cache (cold).",
    "guarded_warm": "StaleGuard, second pass over the same answers: verdict cache warm (repeated questions).",
}


def render_summary(metrics: list[ConfigMetrics], results: list[ItemResult], meta: dict[str, str]) -> str:
    """Markdown report for eval/results/summary.md (linked from the README)."""
    lines = ["# StaleGuard eval results", ""]
    lines += [f"- **{k}:** {v}" for k, v in meta.items()]
    lines += ["", "## Configurations", ""]
    lines += [f"- `{m.config}`: {CONFIG_DESCRIPTIONS.get(m.config, '')}" for m in metrics]
    lines += [
        "",
        "## Results",
        "",
        "| Config | n | Stale recall | Stale precision | False-alarm rate | Median added latency | Tokens / answer | Tavily credits / answer | Tavily credits total | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        lines.append(
            f"| `{m.config}` | {m.n} | {_fmt(m.recall, pct=True)} | {_fmt(m.precision, pct=True)} | "
            f"{_fmt(m.false_alarm_rate, pct=True)} | {_fmt(m.median_latency_ms, digits=0)} ms | "
            f"{_fmt(m.tokens_per_answer, digits=0)} | {_fmt(m.tavily_calls_per_answer, digits=2)} | {m.tavily_credits} | {m.errors} |"
        )
    lines += ["", "Confusion counts (TP / FP / FN / TN): " + "; ".join(f"`{m.config}` {m.tp}/{m.fp}/{m.fn}/{m.tn}" for m in metrics)]

    misses = [r for r in results if r.predicted is not None and r.predicted != r.label]
    if misses:
        lines += ["", "## Misclassified items", "", "| Config | Item | Label | Predicted | Flagged claims |", "|---|---|---|---|---|"]
        for r in misses:
            flagged = "; ".join(r.flagged_claims).replace("|", "\\|") or "-"
            lines.append(f"| `{r.config}` | {r.item_id} | {r.label} | {r.predicted} | {flagged} |")
    errors = [r for r in results if r.error]
    if errors:
        lines += ["", "## Errors", ""] + [f"- `{r.config}` {r.item_id}: {r.error}" for r in errors]
    return "\n".join(lines) + "\n"
