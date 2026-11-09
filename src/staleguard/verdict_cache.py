"""Verdict cache (repository layer): SQLite store of past claim verdicts with a per-class TTL.

A hit skips Tavily and the judge for that claim. Key = hash of the normalized claim plus the
trusted-domain policy (different domains -> different evidence -> different verdict).

Normalization is a *word set*: the extractor rewords the same fact between runs ("The CEO of OpenAI is
Sam Altman" / "OpenAI's CEO is Sam Altman"), so order, articles, copulas and possessives are ignored.
Numbers are kept exact ("3.12" != "312") and negations are kept. Known limitation: claims that differ
only in word order ("A acquired B" / "B acquired A") share a key.

TTL policy:
- class TTL from SPEC §2 (schemas.CLASS_TTL); `static` is never stored (never verified anyway).
- UNVERIFIABLE verdicts are kept at most 1 day so weak-evidence claims get retried soon, but repeated
  questions still don't burn a Tavily credit every time.
- Verdicts produced after a failed search are never stored (the pipeline's cache_store step skips them).
"""

import hashlib
import re
import sqlite3
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from staleguard.schemas import CLASS_TTL, ClaimClass, ClaimVerdict, Verdict

__all__ = ["UNVERIFIABLE_MAX_TTL", "VerdictCache", "cache_key", "normalize_claim", "ttl_for"]

UNVERIFIABLE_MAX_TTL = timedelta(days=1)

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_POSSESSIVE = re.compile(r"(?<=\w)'s\b")
# Numbers keep inner separators (3.12, 1,000); words keep inner hyphens/apostrophes (gpt-4o, o'brien).
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W_]+(?:[-'][^\W_]+)*")
# Deliberately small: never includes negations (not, no, never) or volatility words (latest, current).
_STOPWORDS = frozenset(
    "a an the of is are was were be been being to in on at for by with as it its this that these those "
    "and or has have had which who whose".split()
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    key         TEXT PRIMARY KEY,
    claim_text  TEXT NOT NULL,
    claim_class TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
)
"""


def normalize_claim(text: str) -> str:
    """Sorted set of content tokens (see module docstring for why a word set)."""
    t = _POSSESSIVE.sub("", unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold())
    tokens = {tok for tok in _TOKEN.findall(t) if tok not in _STOPWORDS}
    return " ".join(sorted(tokens))


def cache_key(text: str, include_domains: list[str] | None = None) -> str:
    domains = ",".join(sorted({d.strip().lower() for d in include_domains or [] if d.strip()}))
    return hashlib.sha256(f"{normalize_claim(text)}\x1f{domains}".encode()).hexdigest()


def ttl_for(claim_class: ClaimClass, verdict: Verdict) -> timedelta | None:
    """How long a verdict stays fresh. None = do not store."""
    ttl = CLASS_TTL[claim_class]
    if ttl is None:
        return None
    return min(ttl, UNVERIFIABLE_MAX_TTL) if verdict is Verdict.UNVERIFIABLE else ttl


def _utcnow() -> datetime:
    return datetime.now(UTC)


class VerdictCache:
    """SQLite-backed verdict cache. Use ':memory:' for tests. Not shared across processes beyond SQLite's own locking."""

    def __init__(self, path: str | Path = ":memory:", now: Callable[[], datetime] = _utcnow) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Single event-loop thread in the app; check_same_thread=False lets TestClient's worker thread use it too.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(_SCHEMA)
        self._db.commit()
        self._now = now

    def get(self, text: str, include_domains: list[str] | None = None) -> ClaimVerdict | None:
        """Fresh cached verdict for this claim, or None. Expired rows are deleted on read."""
        key = cache_key(text, include_domains)
        row = self._db.execute("SELECT payload, expires_at FROM verdicts WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        payload, expires_at = row
        if datetime.fromisoformat(expires_at) <= self._now():
            self._db.execute("DELETE FROM verdicts WHERE key = ?", (key,))
            self._db.commit()
            return None
        return ClaimVerdict.model_validate_json(payload)

    def put(self, verdict: ClaimVerdict, include_domains: list[str] | None = None) -> bool:
        """Store a verdict under its claim text. Returns False when the TTL policy says not to store."""
        claim = verdict.claim
        if claim.claim_class is None:
            return False
        ttl = ttl_for(claim.claim_class, verdict.verdict)
        if ttl is None:
            return False
        now = self._now()
        stored = verdict.model_copy(update={"from_cache": False})
        self._db.execute(
            "INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                cache_key(claim.text, include_domains),
                claim.text,
                claim.claim_class.value,
                verdict.verdict.value,
                stored.model_dump_json(),
                now.isoformat(),
                (now + ttl).isoformat(),
            ),
        )
        self._db.commit()
        return True

    def stats(self) -> dict[str, int]:
        """Row counts per verdict (for /health-style introspection and the eval)."""
        rows = self._db.execute("SELECT verdict, COUNT(*) FROM verdicts GROUP BY verdict").fetchall()
        return {v: n for v, n in rows} | {"total": sum(n for _, n in rows)}

    def clear(self) -> None:
        self._db.execute("DELETE FROM verdicts")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

