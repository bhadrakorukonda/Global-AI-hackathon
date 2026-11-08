"""Stage 3 — gather live web evidence for each volatile claim with Tavily (parallel, one search per claim).

Tavily parameters are verified against tavily-python 0.8.5 and the REST docs, see docs/bifrost-notes.md §3.
"""

import asyncio
import logging
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from staleguard.schemas import Claim, ClaimClass, Evidence, SearchResult

__all__ = ["SearchBackend", "normalize_date", "search_params", "verify_claims"]

log = logging.getLogger(__name__)

# Roles and event states change through news; search the news index with a recency window for them.
_NEWS_CLASSES = {ClaimClass.ROLE, ClaimClass.EVENT_STATUS}
_SNIPPET_CHARS = 600


class SearchBackend(Protocol):
    """Subset of tavily.AsyncTavilyClient used here (AsyncTavilyClient satisfies it as-is)."""

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


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
    if claim.claim_class in _NEWS_CLASSES:
        params |= {"topic": "news", "time_range": "year"}
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


async def _search_one(tavily: SearchBackend, claim: Claim, include_domains: list[str] | None) -> SearchResult:
    query = claim.text
    try:
        raw = await tavily.search(query, **search_params(claim, include_domains))
    except Exception as exc:  # noqa: BLE001 — any search failure degrades to "no evidence" (-> UNVERIFIABLE)
        log.warning("tavily search failed for claim %d: %s", claim.id, exc)
        return SearchResult(claim_id=claim.id, query=query, error=f"{type(exc).__name__}: {exc}")
    return SearchResult(
        claim_id=claim.id,
        query=query,
        evidence=[_to_evidence(r) for r in raw.get("results", []) if r.get("url")],
        answer=raw.get("answer"),
    )


async def verify_claims(
    tavily: SearchBackend, claims: list[Claim], include_domains: list[str] | None = None
) -> list[SearchResult]:
    """One Tavily search per claim, run concurrently. Failures yield empty evidence, never an exception."""
    return list(await asyncio.gather(*(_search_one(tavily, c, include_domains) for c in claims)))
