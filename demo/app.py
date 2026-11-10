"""StaleGuard demo UI (SPEC §7): ask a question, see the model's answer next to what StaleGuard returns.

Needs Bifrost + Redis (docker compose up -d) and the proxy (uv run uvicorn staleguard.main:app --port 8000).
Run:  uv run streamlit run demo/app.py

Each question is ONE call through the proxy; the left pane is what the model (or Bifrost's cache) said,
the right pane is what the client received. Scenario buttons in the sidebar follow the demo script.
"""

import asyncio
import time
from typing import Any

import httpx
import streamlit as st
from guard_view import VERDICT_STYLE, Exchange, ask_guarded, cost_rows, seed_cache

from staleguard.config import get_settings
from staleguard.schemas import ClaimVerdict, Mode

TRAINING_DATA_Q = "What is the latest stable version of Python?"
SEED_Q = "What is the latest stable version of Node.js LTS?"
PARAPHRASE_Q = "Which Node.js release is the current LTS?"
STATIC_Q = "What is the capital of Australia?"
MODES: tuple[Mode, ...] = ("annotate", "rewrite", "block")
MODE_HELP = {
    "annotate": "Return the answer unchanged, with the verification report attached.",
    "rewrite": "Regenerate the answer with the corrected facts and citations.",
    "block": "Withhold the answer when a claim is confidently stale.",
}
REQUEST_TIMEOUT_S = 300  # live verification on free tiers can take 30 s+; Bifrost retries on rate limits

settings = get_settings()
st.set_page_config(page_title="StaleGuard", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- state + actions


def _state() -> dict[str, Any]:
    ss = st.session_state
    ss.setdefault("history", [])  # list of Exchange | dict(kind="seed", ...)
    ss.setdefault("cache_key", f"sg-demo-{int(time.time())}")
    return ss


def run_async(coro_fn, *args, **kwargs):
    """Run one async HTTP call from Streamlit's sync script with its own client."""

    async def go():
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as http:
            return await coro_fn(http, *args, **kwargs)

    return asyncio.run(go())


def ask(question: str, cfg: dict[str, Any]) -> None:
    with st.spinner(f"Asking through StaleGuard ({cfg['mode']})… live web verification takes 10–30 s"):
        ex = run_async(
            ask_guarded,
            cfg["proxy"],
            model=cfg["model"],
            question=question,
            mode=cfg["mode"],
            cache_key=cfg["cache_key"],
            include_domains=cfg["domains"],
        )
    _state()["history"].append(ex)


def seed(question: str, cfg: dict[str, Any]) -> None:
    with st.spinner("Seeding Bifrost's cache directly (no StaleGuard)…"):
        try:
            answer, cache_id = run_async(seed_cache, settings.bifrost_url, model=cfg["model"], question=question,
                                         cache_key=cfg["cache_key"])
        except httpx.HTTPError as exc:
            st.error(f"Couldn't reach Bifrost at {settings.bifrost_url}: {type(exc).__name__}")
            return
        time.sleep(2)  # let Bifrost finish writing the entry before the paraphrase arrives
    _state()["history"].append({"kind": "seed", "question": question, "answer": answer, "cache_id": cache_id})


# ---------------------------------------------------------------- rendering


def proxy_health(proxy: str) -> tuple[bool, str]:
    try:
        r = httpx.get(f"{proxy.rstrip('/')}/health", timeout=3)
        data = r.json()
        return data.get("status") == "ok", f"proxy {data.get('status')}, Bifrost {'up' if data['bifrost']['ok'] else 'down'}"
    except (httpx.HTTPError, ValueError, KeyError):
        return False, "proxy not reachable"


def verdict_badge(verdict: str) -> str:
    color, label = VERDICT_STYLE.get(verdict, ("gray", verdict.title()))
    return f":{color}-badge[{label}]"


def status_line(ex: Exchange) -> str:
    r = ex.report
    parts = []
    if ex.status == "stale" and r:
        n = sum(c.verdict == "STALE" for c in r.claims)
        parts.append(f"{verdict_badge('STALE')} **{n} of {r.claims_checked} checked claims out of date**")
    elif ex.status == "clean" and r and r.claims_checked == 0:
        parts.append(":green-badge[Fast path] **no time-sensitive claims, nothing to search**")
    elif ex.status == "clean":
        parts.append(f"{verdict_badge('CURRENT')} **all checked claims are current**")
    elif ex.status == "unverified":
        parts.append(f"{verdict_badge('UNVERIFIABLE')} **some claims couldn't be confirmed**")
    else:
        parts.append(":gray-badge[Skipped] **not verified**")
    if r and r.action != "none":
        parts.append(f":violet-badge[{r.action}]")
    if ex.cache:
        if ex.cache.cache_hit:
            sim = f", similarity {ex.cache.similarity:.2f}" if ex.cache.similarity else ""
            parts.append(f":blue-badge[Bifrost cache hit: {ex.cache.hit_type}{sim}]")
        else:
            parts.append(":gray-badge[Bifrost cache miss]")
    if r and r.cache_evicted:
        # On a hit the evicted entry is the one that served this answer; on a miss it is the entry Bifrost just wrote.
        hit = ex.cache is not None and ex.cache.cache_hit
        parts.append(":orange-badge[stale cache entry evicted]" if hit else ":orange-badge[stale answer kept out of cache]")
    return " · ".join(parts)


def render_claim(c: ClaimVerdict) -> None:
    with st.container(border=True):
        cached = " · :gray[verdict cache]" if c.from_cache else ""
        cls = c.claim.claim_class.value if c.claim.claim_class else "?"
        st.markdown(f"{verdict_badge(c.verdict.value)} {c.claim.text}  \n"
                    f":gray[{cls} · confidence {c.confidence:.2f}]{cached}")
        if c.corrected_claim:
            st.markdown(f"**Now:** {c.corrected_claim}")
        if c.rationale:
            st.caption(c.rationale)
        for e in c.evidence:
            date = f" · {e.published_date}" if e.published_date else ""
            st.markdown(f"- [{e.title or e.url}]({e.url}){date}")


def render_exchange(ex: Exchange) -> None:
    with st.chat_message("user"):
        st.markdown(ex.question)
    with st.chat_message("assistant", avatar="🛡️"):
        if ex.http_status != 200 and ex.error:
            st.error(ex.error)
            return
        st.markdown(status_line(ex))
        if ex.error:
            st.warning(f"Verification problem, answer passed through unchanged: {ex.error}")
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("##### Model answer")
            st.caption("From Bifrost's cache" if ex.cache and ex.cache.cache_hit else "Straight from the model")
            st.markdown(ex.model_answer or "_(empty)_")
        with right:
            st.markdown(f"##### What StaleGuard returned · `{ex.mode}`")
            r = ex.report
            if r and r.action == "blocked":
                st.error(ex.final_answer)
            elif r and r.action == "rewritten":
                st.success(ex.final_answer)
            else:
                st.markdown(ex.final_answer or "_(empty)_")
                stale = [c for c in (r.claims if r else []) if c.corrected_claim]
                if stale:
                    st.warning("**Out of date:**\n" + "\n".join(f"- {c.corrected_claim}" for c in stale))
        r = ex.report
        if r and r.claims:
            st.markdown("##### Claims checked")
            cols = st.columns(min(len(r.claims), 3))
            for i, c in enumerate(r.claims):
                with cols[i % len(cols)]:
                    render_claim(c)
        if r:
            with st.expander("Cost of this check", expanded=False):
                rows = cost_rows(r)
                if ex.upstream_ms is not None:
                    rows.insert(1, ("Model / cache time", f"{ex.upstream_ms:,} ms"))
                st.markdown("\n".join(f"- **{k}:** {v}" for k, v in rows))
            st.caption(f"Added {r.latency_ms / 1000:.1f} s · {r.tavily_calls} Tavily searches · "
                       f"{sum(r.tokens_by_stage.values()):,} verification tokens · {r.verdict_cache_hits} verdict-cache hits")


def render_seed(item: dict[str, Any]) -> None:
    with st.chat_message("user", avatar="🗄️"):
        st.markdown(f"**Seeded Bifrost's cache directly (no StaleGuard):** {item['question']}")
        st.caption(f"Cached as `{item['cache_id']}`. Stands in for an answer cached weeks ago.")
        st.markdown(item["answer"])


# ---------------------------------------------------------------- page


def sidebar() -> dict[str, Any]:
    ss = _state()
    with st.sidebar:
        st.title("🛡️ StaleGuard")
        st.caption("Catches LLM answers that were true once but aren't anymore, in front of the Bifrost gateway.")
        proxy = st.text_input("StaleGuard URL", "http://localhost:8000")
        ok, detail = proxy_health(proxy)
        (st.success if ok else st.error)(detail, icon="✅" if ok else "⚠️")

        mode = st.radio("Mode", MODES, horizontal=True, help="Sent per request as the x-sg-mode header.")
        st.caption(MODE_HELP[mode])
        model = st.text_input("Answering model", settings.small_model, help="Goes to Bifrost as the request's model.")
        use_cache = st.toggle("Bifrost semantic cache", value=True, help="Sends x-bf-cache-key so Bifrost may answer from cache.")
        if use_cache:
            c1, c2 = st.columns([3, 1])
            c1.code(ss["cache_key"], language=None)
            if c2.button("New", help="Fresh cache key: forget everything cached in this session"):
                ss["cache_key"] = f"sg-demo-{int(time.time())}"
                st.rerun()
        domains = st.text_input("Trusted domains (optional)", placeholder="python.org, nodejs.org",
                                help="Tavily prefers these sources (x-sg-include-domains).")
        cfg = {
            "proxy": proxy,
            "mode": mode,
            "model": model,
            "cache_key": ss["cache_key"] if use_cache else None,
            "domains": [d.strip() for d in domains.split(",") if d.strip()] or None,
        }

        st.divider()
        st.subheader("Demo script")
        if st.button("1 · Stale from training data", width="stretch", help=TRAINING_DATA_Q):
            cfg["action"] = ("ask", TRAINING_DATA_Q)
        if st.button("2 · Stale from cache", width="stretch", disabled=not use_cache,
                     help=f"Seed Bifrost with “{SEED_Q}”, then ask “{PARAPHRASE_Q}” through StaleGuard."):
            cfg["action"] = ("cache", None)
        if st.button("2b · Ask again after eviction", width="stretch", disabled=not use_cache, help=PARAPHRASE_Q):
            cfg["action"] = ("ask", PARAPHRASE_Q)
        if st.button("3 · No false alarm", width="stretch", help=STATIC_Q):
            cfg["action"] = ("ask", STATIC_Q)
        if st.button("Clear conversation", width="stretch", type="tertiary"):
            ss["history"] = []
            st.rerun()
    return cfg


def main() -> None:
    cfg = sidebar()
    ss = _state()
    if not ss["history"]:
        st.markdown("### Ask something whose answer changes over time")
        st.caption("Try a library version, a model price, a CEO, or a regulation deadline. Or use the demo script on the left.")

    for item in ss["history"]:
        if isinstance(item, dict):
            render_seed(item)
        else:
            render_exchange(item)

    question = st.chat_input("Ask anything…")
    action = ("ask", question) if question else cfg.get("action")
    if not action:
        return
    kind, q = action
    if kind == "cache":
        seed(SEED_Q, cfg)
        ask(PARAPHRASE_Q, cfg)
    else:
        ask(q, cfg)
    st.rerun()


main()
