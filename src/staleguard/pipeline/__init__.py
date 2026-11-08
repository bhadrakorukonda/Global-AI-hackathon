"""Verification pipeline: extract -> classify -> verify (Tavily) -> judge. `act` (rewrite/block/evict) is step 5."""

from staleguard.pipeline.graph import Pipeline, PipelineDeps

__all__ = ["Pipeline", "PipelineDeps"]
