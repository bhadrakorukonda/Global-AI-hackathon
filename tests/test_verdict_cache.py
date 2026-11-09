"""Verdict cache TTL logic (CLAUDE.md: must be unit-tested). Clock is injected; no sleeping."""

from datetime import UTC, datetime, timedelta

import pytest

from staleguard.schemas import Claim, ClaimClass, ClaimVerdict, Verdict
from staleguard.verdict_cache import UNVERIFIABLE_MAX_TTL, VerdictCache, cache_key, normalize_claim, ttl_for

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def verdict(text: str, cls: ClaimClass, v: Verdict = Verdict.CURRENT, claim_id: int = 1) -> ClaimVerdict:
    return ClaimVerdict(claim=Claim(id=claim_id, text=text, claim_class=cls), verdict=v, confidence=0.9, rationale="r")


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cache(clock: Clock) -> VerdictCache:
    c = VerdictCache(":memory:", now=clock)
    yield c
    c.close()


# ------------------------------------------------------------------ ttl policy


@pytest.mark.parametrize(
    ("cls", "v", "expected"),
    [
        (ClaimClass.PRICE, Verdict.STALE, timedelta(days=1)),
        (ClaimClass.VERSION, Verdict.CURRENT, timedelta(days=7)),
        (ClaimClass.POLICY, Verdict.CONFLICTING, timedelta(days=14)),
        (ClaimClass.ROLE, Verdict.CURRENT, timedelta(days=30)),
        (ClaimClass.ROLE, Verdict.UNVERIFIABLE, UNVERIFIABLE_MAX_TTL),  # capped
        (ClaimClass.PRICE, Verdict.UNVERIFIABLE, timedelta(days=1)),  # class TTL already <= cap
        (ClaimClass.STATIC, Verdict.CURRENT, None),
    ],
)
def test_ttl_for(cls, v, expected):
    assert ttl_for(cls, v) == expected


# ------------------------------------------------------------------ expiry


@pytest.mark.parametrize(
    ("cls", "ttl"),
    [(ClaimClass.PRICE, timedelta(days=1)), (ClaimClass.VERSION, timedelta(days=7)), (ClaimClass.ROLE, timedelta(days=30))],
)
def test_entry_fresh_until_class_ttl_then_expires(cache, clock, cls, ttl):
    assert cache.put(verdict("claim", cls))
    clock.advance(ttl - timedelta(seconds=1))
    assert cache.get("claim") is not None
    clock.advance(timedelta(seconds=1))
    assert cache.get("claim") is None


def test_expired_row_is_deleted_on_read(cache, clock):
    cache.put(verdict("claim", ClaimClass.PRICE))
    clock.advance(timedelta(days=2))
    assert cache.get("claim") is None
    assert cache.stats()["total"] == 0


def test_unverifiable_expires_after_one_day_even_for_long_ttl_class(cache, clock):
    cache.put(verdict("Sam Altman is CEO of OpenAI", ClaimClass.ROLE, Verdict.UNVERIFIABLE))
    clock.advance(timedelta(hours=23))
    assert cache.get("Sam Altman is CEO of OpenAI") is not None
    clock.advance(timedelta(hours=1))
    assert cache.get("Sam Altman is CEO of OpenAI") is None


def test_static_and_unclassified_are_not_stored(cache):
    assert not cache.put(verdict("Water boils at 100C", ClaimClass.STATIC))
    unclassified = ClaimVerdict(claim=Claim(id=1, text="x"), verdict=Verdict.CURRENT, confidence=1.0)
    assert not cache.put(unclassified)
    assert cache.stats()["total"] == 0


def test_rewrite_refreshes_expiry(cache, clock):
    cache.put(verdict("claim", ClaimClass.PRICE))
    clock.advance(timedelta(hours=20))
    cache.put(verdict("claim", ClaimClass.PRICE, Verdict.STALE))
    clock.advance(timedelta(hours=20))  # 40h after first put, 20h after second
    hit = cache.get("claim")
    assert hit is not None and hit.verdict is Verdict.STALE


# ------------------------------------------------------------------ keys


def test_normalization_matches_trivial_variants(cache):
    cache.put(verdict("The latest stable version of Python is 3.12.", ClaimClass.VERSION))
    assert cache.get("  the LATEST stable   version of python is 3.12 ") is not None
    assert cache.get("The latest stable version of Python is 312") is None  # inner punctuation matters


def test_normalize_claim_quotes_and_unicode():
    assert normalize_claim("India’s population is 1.42 billion!") == "1.42 billion india population"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("The CEO of OpenAI is Sam Altman.", "OpenAI's CEO is Sam Altman"),  # seen live: extractor rewording
        ("GPT-4 is the newest model of OpenAI.", "GPT-4 is OpenAI's newest model."),
    ],
)
def test_rewordings_share_a_key(a, b):
    assert cache_key(a) == cache_key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Python 3.12 is the latest version.", "Python 3.13 is the latest version."),
        ("The merger is approved.", "The merger is not approved."),
        ("GPT-4o costs $5 per million tokens.", "GPT-4o costs $15 per million tokens."),
        ("Python 3.12 is the latest version.", "Python 3.12 is a version."),  # 'latest' is not a stopword
    ],
)
def test_meaningful_differences_get_different_keys(a, b):
    assert cache_key(a) != cache_key(b)


def test_domain_policy_is_part_of_key(cache):
    cache.put(verdict("claim", ClaimClass.POLICY), include_domains=["who.int"])
    assert cache.get("claim") is None
    assert cache.get("claim", include_domains=["WHO.int "]) is not None
    assert cache_key("c", ["a.com", "b.com"]) == cache_key("c", ["b.com", "a.com"])


def test_persists_across_instances(tmp_path, clock):
    path = tmp_path / "sub" / "v.sqlite3"
    first = VerdictCache(path, now=clock)
    first.put(verdict("claim", ClaimClass.ROLE))
    first.close()
    second = VerdictCache(path, now=clock)
    assert second.get("claim") is not None
    second.close()
