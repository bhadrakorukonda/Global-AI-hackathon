"""Pipeline tests: LLM and Tavily faked; Tavily replies are recorded live responses (tests/fixtures/tavily)."""

import asyncio
import json
from datetime import date

import pytest
from fakes import FakeLLM, FakeTavily, load_fixture

from staleguard.llm import LLMOutputError, UsageTracker, structured_call
from staleguard.pipeline import Pipeline, PipelineDeps
from staleguard.pipeline.extract import strip_code
from staleguard.pipeline.judge import JUDGE_SNIPPET_CHARS
from staleguard.pipeline.verify import normalize_date, search_params, search_query
from staleguard.schemas import Claim, ClaimClass, ClaimVerdict, ExtractLLMOutput, Verdict, VerifyRequest
from staleguard.verdict_cache import VerdictCache

TODAY = date(2026, 11, 9)
PY_CLAIM = "The latest stable version of Python is 3.12."
CEO_CLAIM = "Sam Altman is the CEO of OpenAI."


def make_pipeline(llm: FakeLLM, tavily: FakeTavily, max_claims: int = 5) -> Pipeline:
    return Pipeline(
        PipelineDeps(llm=llm, tavily=tavily, small_model="small", judge_model="judge", max_claims=max_claims, today=lambda: TODAY)
    )


def run(pipeline: Pipeline, text: str, **kw) -> object:
    return asyncio.run(pipeline.run(VerifyRequest(text=text, **kw)))


def test_stale_claim_end_to_end():
    llm = FakeLLM(
        extract=[{"claims": [PY_CLAIM, "Python was created by Guido van Rossum."]}],
        classify=[{"items": [{"id": 2, "claim_class": "static"}]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "STALE", "corrected_claim": "The latest stable Python is 3.15.",
                              "evidence_ids": [1, 1, 9], "confidence": 0.9, "rationale": "Newer release exists."}]}],
    )
    tavily = FakeTavily({PY_CLAIM: load_fixture("tavily/general_python_version.json")})
    report = run(make_pipeline(llm, tavily), "Python 3.12 is the latest. Guido made it.")

    assert report.verdict_summary == "stale"
    assert report.claims_extracted == 2 and report.claims_checked == 1
    [v] = report.claims
    assert v.verdict is Verdict.STALE
    assert v.corrected_claim == "The latest stable Python is 3.15."
    # evidence is newest first, so [1] is the 2026 Wikipedia page; the dup and out-of-range ids are dropped
    assert [e.url for e in v.evidence] == ["https://en.wikipedia.org/wiki/History_of_Python"]
    assert v.evidence[0].published_date == "2026-08-08"
    assert v.claim.classified_by == "rules"
    assert llm.stages() == ["extract", "classify", "judge"]
    assert [m for _, m, _ in llm.calls] == ["small", "small", "judge"]  # model tiering
    assert report.tavily_calls == 1
    assert report.tokens_by_stage == {"extract": 100, "classify": 100, "judge": 100}


def test_fast_path_skips_tavily_and_judge_for_static_answer():
    llm = FakeLLM(
        extract=[{"claims": ["Water boils at 100 degrees Celsius at sea level."]}],
        classify=[{"items": [{"id": 1, "claim_class": "static"}]}],
    )
    tavily = FakeTavily()
    report = run(make_pipeline(llm, tavily), "Water boils at 100°C at sea level.")
    assert report.verdict_summary == "clean"
    assert report.claims == [] and report.claims_checked == 0
    assert tavily.calls == []
    assert llm.stages() == ["extract", "classify"]


def test_extractor_class_makes_fast_path_a_single_llm_call():
    llm = FakeLLM(extract=[{"claims": [{"claim": "Canberra is the capital of Australia.", "query": "capital of Australia",
                                         "class": "static"}]}])
    tavily = FakeTavily()
    report = run(make_pipeline(llm, tavily), "The capital of Australia is Canberra.")
    assert llm.stages() == ["extract"] and tavily.calls == []  # no classify call, no search
    assert report.verdict_summary == "clean" and report.claims_checked == 0


def test_rules_override_the_extractor_class_and_missing_classes_fall_back_to_the_classify_call():
    llm = FakeLLM(
        extract=[{"claims": [{"claim": PY_CLAIM, "query": "latest stable Python version", "class": "static"},
                             {"claim": "The Eiffel Tower is in Paris.", "class": "not-a-class"}]}],
        classify=[{"items": [{"id": 2, "claim_class": "static"}]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "STALE", "corrected_claim": "3.15", "evidence_ids": [1], "confidence": 0.9}]}],
    )
    tavily = FakeTavily(default=load_fixture("tavily/general_python_version.json"))
    report = run(make_pipeline(llm, tavily), "...")
    assert llm.stages() == ["extract", "classify", "judge"]
    sent = llm.calls[1][2][0]["content"].rsplit("(JSON):\n", 1)[1]
    assert json.loads(sent) == [{"id": 2, "text": "The Eiffel Tower is in Paris."}]  # only the claim without a valid class
    (v,) = report.claims
    assert v.claim.claim_class is ClaimClass.VERSION and v.claim.classified_by == "rules"


@pytest.mark.parametrize(("raw", "expected"), [("Version ", ClaimClass.VERSION), ("static", ClaimClass.STATIC),
                                               ("weird", None), (None, None), (3, None)])
def test_extracted_class_is_parsed_leniently(raw, expected):
    out = ExtractLLMOutput.model_validate({"claims": [{"claim": "c", "class": raw}]})
    assert out.claims[0].claim_class is expected


def test_no_claims_skips_everything_after_extract():
    llm = FakeLLM(extract=[{"claims": []}])
    report = run(make_pipeline(llm, FakeTavily()), "Happy to help!")
    assert report.verdict_summary == "clean"
    assert llm.stages() == ["extract"]


def test_code_only_answer_makes_no_llm_calls():
    llm = FakeLLM()
    report = run(make_pipeline(llm, FakeTavily()), "```python\nprint('hi')\n```")
    assert report.claims_extracted == 0
    assert llm.calls == []


def test_verdict_without_cited_evidence_is_downgraded_to_unverifiable():
    llm = FakeLLM(
        extract=[{"claims": [CEO_CLAIM]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "STALE", "corrected_claim": "X", "evidence_ids": [], "confidence": 0.95}]}],
    )
    tavily = FakeTavily({CEO_CLAIM: load_fixture("tavily/news_openai_ceo.json")})
    report = run(make_pipeline(llm, tavily), "Sam Altman runs OpenAI as CEO.")
    [v] = report.claims
    assert v.verdict is Verdict.UNVERIFIABLE
    assert v.corrected_claim is None and v.confidence == 0.0
    assert report.verdict_summary == "unverified"


def test_no_evidence_is_unverifiable_without_calling_judge():
    llm = FakeLLM(extract=[{"claims": [CEO_CLAIM]}])
    report = run(make_pipeline(llm, FakeTavily(fail=True)), "Sam Altman is OpenAI's CEO.")
    [v] = report.claims
    assert v.verdict is Verdict.UNVERIFIABLE
    assert "TimeoutError" in v.rationale
    assert "judge" not in llm.stages()


def test_current_claim_is_clean_and_news_topic_used_for_roles():
    llm = FakeLLM(
        extract=[{"claims": [CEO_CLAIM]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "CURRENT", "evidence_ids": [1, 3], "confidence": 0.85}]}],
    )
    tavily = FakeTavily({CEO_CLAIM: load_fixture("tavily/news_openai_ceo.json")})
    report = run(make_pipeline(llm, tavily), "Sam Altman is OpenAI's CEO.", include_domains=["openai.com"])
    assert report.verdict_summary == "clean"
    [(query, kwargs)] = tavily.calls
    assert query == "Sam Altman is the CEO of OpenAI 2026"  # no extractor query -> claim text + year
    assert kwargs["topic"] == "news" and kwargs["time_range"] == "year"
    assert kwargs["include_domains"] == ["openai.com"] and kwargs["include_domains_mode"] == "prefer"


def test_claim_cap_limits_tavily_calls():
    claims = [f"Product {i} costs ${i} per month." for i in range(1, 8)]
    verdicts = [{"id": i, "verdict": "UNVERIFIABLE", "confidence": 0.1} for i in range(1, 8)]
    llm = FakeLLM(extract=[{"claims": claims}], judge=[{"verdicts": verdicts}])
    tavily = FakeTavily(default=load_fixture("tavily/general_python_version.json"))
    report = run(make_pipeline(llm, tavily, max_claims=3), "prices...")
    assert len(tavily.calls) == 3 and report.claims_checked == 3


def _cached_pipeline(llm: FakeLLM, tavily: FakeTavily, cache: VerdictCache) -> Pipeline:
    return Pipeline(PipelineDeps(llm=llm, tavily=tavily, small_model="s", judge_model="j", verdict_cache=cache, today=lambda: TODAY))


CEO_JUDGE = {"verdicts": [{"id": 1, "verdict": "CURRENT", "evidence_ids": [1], "confidence": 0.9}]}


def test_second_run_is_served_from_verdict_cache():
    cache = VerdictCache(":memory:")
    tavily = FakeTavily({CEO_CLAIM: load_fixture("tavily/news_openai_ceo.json")})
    llm = FakeLLM(extract=[{"claims": [CEO_CLAIM]}, {"claims": ["sam altman is the CEO of OpenAI"]}], judge=[CEO_JUDGE])
    pipeline = _cached_pipeline(llm, tavily, cache)

    first = run(pipeline, "Sam Altman is OpenAI's CEO.")
    assert first.tavily_calls == 1 and first.verdict_cache_hits == 0
    assert not first.claims[0].from_cache

    second = run(pipeline, "Sam Altman is OpenAI's CEO.")
    assert second.tavily_calls == 0 and second.verdict_cache_hits == 1
    [v] = second.claims
    assert v.from_cache and v.verdict is Verdict.CURRENT
    assert v.claim.text == "sam altman is the CEO of OpenAI"  # this answer's wording, cached judgement
    assert len(tavily.calls) == 1
    assert llm.stages() == ["extract", "judge", "extract"]  # no second judge call
    assert "judge" not in second.tokens_by_stage


def test_search_failures_are_not_cached():
    cache = VerdictCache(":memory:")
    llm = FakeLLM(extract=[{"claims": [CEO_CLAIM]}, {"claims": [CEO_CLAIM]}])
    tavily = FakeTavily(fail=True)
    pipeline = _cached_pipeline(llm, tavily, cache)
    run(pipeline, "x")
    second = run(pipeline, "x")
    assert second.verdict_cache_hits == 0 and len(tavily.calls) == 2
    assert cache.stats()["total"] == 0


def test_mixed_cached_and_fresh_claims_are_merged_in_order():
    cache = VerdictCache(":memory:")
    cache.put(ClaimVerdict(claim=Claim(id=9, text=CEO_CLAIM, claim_class=ClaimClass.ROLE), verdict=Verdict.CURRENT, confidence=0.9))
    llm = FakeLLM(
        extract=[{"claims": [PY_CLAIM, CEO_CLAIM]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "STALE", "corrected_claim": "3.14", "evidence_ids": [1], "confidence": 0.9}]}],
    )
    tavily = FakeTavily(default=load_fixture("tavily/general_python_version.json"))
    report = run(_cached_pipeline(llm, tavily, cache), "...")
    assert [(v.claim.id, v.from_cache) for v in report.claims] == [(1, False), (2, True)]
    assert [q for q, _ in tavily.calls] == ["The latest stable version of Python is 3.12 2026"]
    assert report.verdict_summary == "stale"


def test_search_params_for_claim_about_now():
    p = search_params(Claim(id=1, text="The latest Python is 3.12", claim_class=ClaimClass.VERSION), None)
    assert p == {"search_depth": "basic", "max_results": 5, "include_answer": True, "include_published_date": True,
                 "time_range": "year"}
    assert "days" not in p


def test_search_params_for_time_anchored_claim_have_no_recency_filter():
    role = Claim(id=1, text="Jack Dorsey was CEO of Twitter in 2015.", claim_class=ClaimClass.ROLE)
    assert "time_range" not in search_params(role, None) and "topic" not in search_params(role, None)


@pytest.mark.parametrize(
    ("text", "query", "expected"),
    [
        ("The latest stable version of Python is 3.12.", "latest stable Python version", "latest stable Python version 2026"),
        ("Python 3.12 was released in October 2023.", "Python 3.12 release date", "Python 3.12 release date"),
        ("Sam Altman is the CEO of OpenAI.", None, "Sam Altman is the CEO of OpenAI 2026"),
        ("The latest Node.js LTS is 20.", "Node.js LTS 2026", "Node.js LTS 2026"),  # no double year
    ],
)
def test_search_query(text, query, expected):
    assert search_query(Claim(id=1, text=text, search_query=query), TODAY) == expected


def test_extractor_query_is_used_and_evidence_is_newest_first():
    llm = FakeLLM(
        extract=[{"claims": [{"claim": PY_CLAIM, "query": "latest stable Python version"}]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "STALE", "corrected_claim": "3.15", "evidence_ids": [1], "confidence": 0.9}]}],
    )
    tavily = FakeTavily(default=load_fixture("tavily/general_python_version.json"))
    report = run(make_pipeline(llm, tavily), "Python 3.12 is the latest.")
    assert tavily.calls[0][0] == "latest stable Python version 2026"
    evidence = judge_payload(llm)[0]["evidence"]
    dates = [e["published"] for e in evidence if e["published"]]
    assert len(dates) >= 2 and dates == sorted(dates, reverse=True)
    assert report.claims[0].search_query == "latest stable Python version 2026"


def judge_payload(llm: FakeLLM) -> list[dict]:
    """The claims+evidence JSON the judge prompt was rendered with."""
    prompt = llm.calls[-1][2][0]["content"]
    return json.loads(prompt.rsplit("(JSON):\n", 1)[1])


def test_judge_input_is_trimmed_but_report_keeps_full_evidence():
    llm = FakeLLM(
        extract=[{"claims": [PY_CLAIM]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "STALE", "corrected_claim": "3.15", "evidence_ids": [1], "confidence": 0.9}]}],
    )
    tavily = FakeTavily(default=load_fixture("tavily/general_python_version.json"))
    report = run(make_pipeline(llm, tavily), "Python 3.12 is the latest.")
    sent = judge_payload(llm)[0]["evidence"]
    assert all(len(e["snippet"]) <= JUDGE_SNIPPET_CHARS and "url" not in e for e in sent)
    assert all("/" not in e["source"] and not e["source"].startswith("www.") for e in sent)
    cited = report.claims[0].evidence[0]
    assert cited.url.startswith("https://") and len(cited.snippet) > JUDGE_SNIPPET_CHARS  # full evidence kept for the UI


@pytest.mark.parametrize(
    ("raw", "iso"),
    [("Fri, 04 Jul 2025 20:09:07 GMT", "2025-07-04"), ("2026-03-01T10:00:00Z", "2026-03-01"), (None, None), ("garbage", None)],
)
def test_normalize_date(raw, iso):
    assert normalize_date(raw) == iso


def test_strip_code_removes_fences_keeps_inline_text():
    assert strip_code("Use `pip`.\n```bash\npip install x\n```\nDone.") == "Use pip.\n \nDone."


class _Backend:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.messages: list[list[dict[str, str]]] = []

    async def complete(self, model, messages, reasoning_effort):
        from staleguard.llm import LLMCompletion

        self.messages.append(messages)
        return LLMCompletion(text=self.texts.pop(0), total_tokens=7)


def _call(backend: _Backend, usage: UsageTracker):
    return asyncio.run(
        structured_call(backend, model="m", prompt="extract", variables={"question": "", "text": "t"},
                        output_model=ExtractLLMOutput, stage="extract", usage=usage)
    )


def test_structured_call_retries_once_then_succeeds():
    backend, usage = _Backend("not json", '```json\n{"claims": ["a"]}\n```'), UsageTracker()
    assert [c.claim for c in _call(backend, usage).claims] == ["a"]
    assert len(backend.messages) == 2 and "not valid" in backend.messages[1][-1]["content"]
    assert usage.snapshot() == {"extract": 14}


def test_extract_output_accepts_bare_list_and_bare_strings():
    out = ExtractLLMOutput.model_validate([{"claim": "a", "query": "q"}, "b"])
    assert [(c.claim, c.query) for c in out.claims] == [("a", "q"), ("b", None)]


def test_structured_call_gives_up_after_second_failure():
    with pytest.raises(LLMOutputError):
        _call(_Backend("nope", '{"wrong": 1}'), UsageTracker())
