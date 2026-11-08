# CLAUDE.md

Read `SPEC.md` first. It is the source of truth for scope, architecture and build order.

## Context
- Hackathon prototype, deadline **15 Nov 2026, 11:59 PM IST**. Ship a working, demoable slice over a broad, half-built one.
- Judges will read the README, watch a demo video and skim the repo. A clean setup that actually runs matters.

## How to work
- Follow the build order in SPEC.md §9. After each step the repo must run end to end; commit at each step.
- Before relying on any Bifrost behaviour (cache headers, `cache_debug` fields, cache eviction endpoint, config schema), confirm it against the current Bifrost docs or source and note what you found in `docs/bifrost-notes.md`. Same for Tavily SDK parameters.
- Ask the user before: changing the architecture, adding a new external service, or anything that spends paid API credits in bulk (e.g. running the full eval).
- Don't fabricate eval data. Scaffold the dataset format and a few examples, then hand it back to the user to fill and verify.

## Conventions
- Python 3.11, `uv`, type hints, Pydantic v2 models for every pipeline input/output.
- All LLM calls go through Bifrost via the `openai` SDK (`base_url=BIFROST_URL/v1`). Never call providers directly.
- Prompts live in `src/staleguard/prompts/*.md`, not inline strings.
- LLM outputs that feed code must be structured JSON, validated with Pydantic, with one retry on parse failure.
- Secrets come only from `.env`; never commit them.
- Tests: pytest, Tavily and LLM calls mocked with recorded fixtures. Unit-test the classifier rules and the verdict cache TTL logic.
