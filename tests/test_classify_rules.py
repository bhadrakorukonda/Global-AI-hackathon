"""Unit tests for the deterministic volatility rules (CLAUDE.md: classifier rules must be unit-tested)."""

from datetime import date, timedelta

import pytest

from staleguard.pipeline.classify import CLASS_TTL, classify_by_rules, select_for_verification
from staleguard.schemas import Claim, ClaimClass

TODAY = date(2026, 11, 9)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # price
        ("GPT-4o costs $5 per million input tokens.", ClaimClass.PRICE),
        ("ChatGPT Plus is priced at 20 USD.", ClaimClass.PRICE),
        ("The plan costs ₹499 per month.", ClaimClass.PRICE),
        # version
        ("The latest stable version of Python is 3.12.", ClaimClass.VERSION),
        ("React 19.0 is the current stable release.", ClaimClass.VERSION),
        ("Node.js v22.11 is the active LTS line.", ClaimClass.VERSION),
        # role
        ("Sam Altman is the CEO of OpenAI.", ClaimClass.ROLE),
        ("Narendra Modi is the Prime Minister of India.", ClaimClass.ROLE),
        ("Satya Nadella is chairman of Microsoft.", ClaimClass.ROLE),
        # event_status
        ("The merger between the two companies is pending regulatory approval.", ClaimClass.EVENT_STATUS),
        ("The feature is in beta.", ClaimClass.EVENT_STATUS),
        # policy
        ("The EU AI Act bans social scoring systems.", ClaimClass.POLICY),
        ("WHO guidelines recommend 150 minutes of exercise a week.", ClaimClass.POLICY),
        # statistic
        ("India's population is 1.42 billion.", ClaimClass.STATISTIC),
        ("ChatGPT has 800 million weekly users.", ClaimClass.STATISTIC),
        ("Chrome holds about 65% of browser market share.", ClaimClass.STATISTIC),
        # relative_time
        ("Python is currently the most popular programming language.", ClaimClass.STATISTIC),  # outranks relative_time
        ("The newest iPhone has a titanium frame.", ClaimClass.RELATIVE_TIME),
        ("Remote work is common nowadays.", ClaimClass.RELATIVE_TIME),
        ("The conference takes place in 2026.", ClaimClass.RELATIVE_TIME),  # recent year alone
    ],
)
def test_rules_classify_volatile_claims(text, expected):
    cls, signals = classify_by_rules(text, TODAY)
    assert cls is expected, signals
    assert signals


@pytest.mark.parametrize(
    "text",
    [
        "Water boils at 100 degrees Celsius at sea level.",
        "Jupiter is the largest planet in the solar system.",
        "Python was created by Guido van Rossum.",
        "The French Revolution began in 1789.",  # old year is not volatile
        "A triangle has three sides.",
        "This approach leads to cleaner code.",  # 'leads' must not fire role
        "The test runs in under a second.",  # 'runs' must not fire role
    ],
)
def test_rules_leave_static_claims_for_llm(text):
    cls, _ = classify_by_rules(text, TODAY)
    assert cls is None


def test_specific_class_wins_over_relative_time():
    cls, signals = classify_by_rules("The current CEO of Apple is Tim Cook.", TODAY)
    assert cls is ClaimClass.ROLE
    assert "relative_word" in signals


def test_last_year_counts_as_recent_but_two_years_ago_does_not():
    assert classify_by_rules("Released in 2025.", TODAY)[0] is ClaimClass.RELATIVE_TIME
    assert classify_by_rules("Released in 2024.", TODAY)[0] is None


def test_every_class_has_a_ttl_and_only_static_is_never():
    assert set(CLASS_TTL) == set(ClaimClass)
    assert CLASS_TTL[ClaimClass.STATIC] is None
    assert CLASS_TTL[ClaimClass.PRICE] == timedelta(days=1)
    assert CLASS_TTL[ClaimClass.ROLE] == timedelta(days=30)


def _claim(i: int, cls: ClaimClass) -> Claim:
    return Claim(id=i, text=f"claim {i}", claim_class=cls)


def test_select_drops_static_and_caps_by_volatility():
    claims = [
        _claim(1, ClaimClass.ROLE),  # 30d
        _claim(2, ClaimClass.STATIC),
        _claim(3, ClaimClass.PRICE),  # 1d
        _claim(4, ClaimClass.VERSION),  # 7d
        _claim(5, ClaimClass.STATISTIC),  # 30d
    ]
    picked = select_for_verification(claims, max_claims=2)
    assert [c.id for c in picked] == [3, 4]  # most volatile, returned in answer order


def test_select_returns_nothing_for_all_static():
    assert select_for_verification([_claim(1, ClaimClass.STATIC)], max_claims=5) == []
