"""Stage 3 — gather live web evidence for each volatile claim with Tavily (parallel, one search per claim).

Query policy (A/B-tested live on 2026-11-10, see docs/bifrost-notes.md §3):
- Query = the extractor's neutral topic query ("latest stable Python version"), never the claim text when
  avoidable: a query that repeats the stale value ("...is 3.12") retrieves pages agreeing with it.
- A claim about *now* (no explicit year in it) gets the current year appended and time_range="year".
  That fixed 3/3 current-value probes (Python, Node.js LTS, OpenAI model) that the claim-text query got wrong.
- A claim anchored to a year ("released in October 2023") is about the past: no year, no recency filter
  (the filter made Tavily invent a 2026 release date for Python 3.12).
- Evidence is ordered newest first (undated last) so the judge reads the freshest sources first.

Tavily parameters are verified against tavily-python 0.8.5 and the REST docs, see docs/bifrost-notes.md §3.
"""

import asyncio
import logging
import re
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from staleguard.schemas import Claim, ClaimClass, Evidence, SearchResult

__all__ = ["SearchBackend", "is_time_anchored", "normalize_date", "search_params", "search_query", "verify_claims"]

log = logging.getLogger(__name__)

# Roles and event states change through news; search the news index with a recency window for them.
_NEWS_CLASSES = {ClaimClass.ROLE, ClaimClass.EVENT_STATUS}
_SNIPPET_CHARS = 600
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


class SearchBackend(Protocol):
    """Subset of tavily.AsyncTavilyClient used here (AsyncTavilyClient satisfies it as-is)."""

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


def is_time_anchored(claim: Claim) -> bool:
    """True when the claim names a year, i.e. it is about a point in the past rather than "now"."""
    return bool(_YEAR.search(claim.text))


def search_query(claim: Claim, today: date) -> str:
    """Neutral topic query (claim text as fallback); current-state claims get the current year appended."""
    query = (claim.search_query or claim.text).strip().rstrip(".")
    if not is_time_anchored(claim) and not _YEAR.search(query):
        query = f"{query} {today.year}"
    return query


def search_params(claim: Claim, include_domains: list[str] | None) -> dict[str, Any]:
    """Tavily search() kwargs for a claim.

    max_results is explicit (SDK default is 10); include_published_date is passed through **kwargs
    because the SDK has no named parameter for it, and it is only automatic for topic=news.
    """
    params: dict[str, Any] = {
        "search_depth": "basic",  # 1 credit; advanced costs 2
        "max_results": 5,
        "include_answer": True,
        "include_published_date": True,
    }
    if not is_time_anchored(claim):
        params["time_range"] = "year"  # claims about "now": only recent sources count
        if claim.claim_class in _NEWS_CLASSES:
            params["topic"] = "news"
    if include_domains:
        params |= {"include_domains": include_domains, "include_domains_mode": "prefer"}
    return params


def normalize_date(value: str | None) -> str | None:
    """Tavily returns RFC 2822 dates ('Fri, 04 Jul 2025 20:09:07 GMT'); convert to YYYY-MM-DD."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError):
        return value[:10] if len(value) >= 10 and value[4] == "-" else None


def _to_evidence(item: dict[str, Any]) -> Evidence:
    return Evidence(
        url=item.get("url", ""),
        title=item.get("title") or "",
        snippet=(item.get("content") or "")[:_SNIPPET_CHARS],
        published_date=normalize_date(item.get("published_date")),
        score=item.get("score"),
    )


def _newest_first(evidence: list[Evidence]) -> list[Evidence]:
    return sorted(evidence, key=lambda e: e.published_date or "", reverse=True)  # ISO dates sort; undated last


async def _search_one(tavily: SearchBackend, claim: Claim, include_domains: list[str] | None, today: date) -> SearchResult:
    query = search_query(claim, today)
    try:
        raw = await tavily.search(query, **search_params(claim, include_domains))
    except Exception as exc:  # noqa: BLE001 — any search failure degrades to "no evidence" (-> UNVERIFIABLE)
        log.warning("tavily search failed for claim %d: %s", claim.id, exc)
        return SearchResult(claim_id=claim.id, query=query, error=f"{type(exc).__name__}: {exc}")
    return SearchResult(
        claim_id=claim.id,
        query=query,
        evidence=_newest_first([_to_evidence(r) for r in raw.get("results", []) if r.get("url")]),
        answer=raw.get("answer"),
    )


async def verify_claims(
    tavily: SearchBackend, claims: list[Claim], today: date, include_domains: list[str] | None = None
) -> list[SearchResult]:
    """One Tavily search per claim, run concurrently. Failures yield empty evidence, never an exception."""
    return list(await asyncio.gather(*(_search_one(tavily, c, include_domains, today) for c in claims)))
