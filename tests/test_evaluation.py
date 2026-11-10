"""Eval scaffolding: dataset validation, answer-level scoring, metrics, summary, verify-all baseline."""

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest
from fakes import FakeLLM, FakeTavily, load_fixture

from staleguard.evaluation import DatasetError, ItemResult, compute_metrics, item_result, load_dataset, render_summary
from staleguard.pipeline import Pipeline, PipelineDeps
from staleguard.schemas import Claim, ClaimClass, ClaimVerdict, Report, Verdict, VerifyRequest

REPO_DATASET = Path(__file__).parent.parent / "eval" / "dataset.jsonl"

ROW = {
    "id": "a", "question": "q", "answer_text": "t", "label": "stale", "class": "version",
    "ground_truth": "g", "source_url": "https://example.com", "verified": True,
}


def write(tmp_path: Path, *rows: dict | str) -> Path:
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_repo_dataset_is_valid():
    items = load_dataset(REPO_DATASET, include_unverified=True)
    assert len(items) >= 4 and {i.label for i in items} == {"stale", "current"}


def test_unverified_rows_are_excluded_by_default(tmp_path):
    path = write(tmp_path, "// comment", ROW, ROW | {"id": "b", "verified": False}, "")
    assert [i.id for i in load_dataset(path)] == ["a"]
    assert [i.id for i in load_dataset(path, include_unverified=True)] == ["a", "b"]


def test_dataset_errors_list_every_bad_line(tmp_path):
    path = write(tmp_path, ROW, ROW, ROW | {"id": "c", "label": "old"}, "{not json", ROW | {"id": "d", "source_url": "nope"})
    with pytest.raises(DatasetError) as exc:
        load_dataset(path)
    msg = str(exc.value)
    assert "line 2: duplicate id 'a'" in msg and "line 3" in msg and "line 4" in msg and "line 5" in msg


def test_unknown_fields_are_rejected(tmp_path):
    with pytest.raises(DatasetError, match="line 1"):
        load_dataset(write(tmp_path, ROW | {"lable": "stale"}))


def _report(*verdicts: tuple[Verdict, float, str | None]) -> Report:
    claims = [ClaimVerdict(claim=Claim(id=i, text=f"c{i}", claim_class=ClaimClass.VERSION), verdict=v, confidence=c,
                           corrected_claim=fix) for i, (v, c, fix) in enumerate(verdicts, start=1)]
    return Report(verdict_summary="stale", claims=claims, claims_extracted=len(claims), claims_checked=len(claims),
                  latency_ms=100, tokens_by_stage={"extract": 10, "judge": 30}, tavily_calls=len(claims))


def test_item_prediction_uses_the_proxy_action_rule(tmp_path):
    item = load_dataset(write(tmp_path, ROW))[0]
    assert item_result(item, "g", _report((Verdict.STALE, 0.9, "new")), 0.7).predicted == "stale"
    assert item_result(item, "g", _report((Verdict.STALE, 0.5, "new")), 0.7).predicted == "current"  # low confidence
    r = item_result(item, "g", _report((Verdict.CURRENT, 0.9, None), (Verdict.STALE, 0.8, "fix")), 0.7)
    assert r.predicted == "stale" and r.flagged_claims == ["c2"] and r.corrections == ["fix"] and r.tokens == 40


def _res(label: str, predicted: str | None, config: str = "g", calls: int = 1, latency: int = 100) -> ItemResult:
    return ItemResult(item_id=f"{label}-{predicted}", config=config, label=label, claim_class=ClaimClass.VERSION,
                      predicted=predicted, tavily_calls=calls, latency_ms=latency, tokens=50,
                      error=None if predicted else "boom")


def test_compute_metrics():
    results = [_res("stale", "stale"), _res("stale", "stale"), _res("stale", "current"),
               _res("current", "current"), _res("current", "stale"), _res("current", "current", latency=300),
               _res("stale", None), _res("stale", "stale", config="other")]
    m = compute_metrics("g", results)
    assert (m.n, m.errors, m.tp, m.fp, m.fn, m.tn) == (7, 1, 2, 1, 1, 2)
    assert m.recall == pytest.approx(2 / 3) and m.precision == pytest.approx(2 / 3)
    assert m.false_alarm_rate == pytest.approx(1 / 3)
    assert m.median_latency_ms == 100 and m.tavily_credits == 6 and m.tavily_calls_per_answer == 1.0


def test_metrics_with_no_positives_are_none_not_zero_division():
    m = compute_metrics("g", [_res("current", "current")])
    assert m.recall is None and m.precision is None and m.false_alarm_rate == 0.0


def test_render_summary_has_table_and_misses():
    results = [_res("stale", "stale"), _res("current", "stale"), _res("stale", None)]
    md = render_summary([compute_metrics("g", results)], results, {"Run": "now"})
    assert "| `g` | 3 | 100% | 50% | 100% |" in md
    assert "## Misclassified items" in md and "current-stale" in md and "## Errors" in md


def test_verify_all_baseline_skips_classify_and_checks_static_claims():
    llm = FakeLLM(
        extract=[{"claims": ["Water boils at 100 degrees Celsius at sea level.", "The latest Python is 3.12."]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "CURRENT", "evidence_ids": [1], "confidence": 0.9},
                             {"id": 2, "verdict": "STALE", "corrected_claim": "3.15", "evidence_ids": [1], "confidence": 0.9}]}],
    )
    tavily = FakeTavily(default=load_fixture("tavily/general_python_version.json"))
    deps = PipelineDeps(llm=llm, tavily=tavily, small_model="s", judge_model="j", volatility_filter=False,
                        today=lambda: date(2026, 11, 9))
    report = asyncio.run(Pipeline(deps).run(VerifyRequest(text="...")))
    assert llm.stages() == ["extract", "judge"]  # no classify call
    assert len(tavily.calls) == 2 and report.claims_checked == 2
