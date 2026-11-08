<!-- classify v1 — model: SMALL_MODEL. Placeholders: {{claims}} -->
Classify each claim by how likely it is to go out of date. Pick exactly one class:

- version: software/product version or release ("The latest Python is 3.12")
- role: who holds a position ("X is the CEO of Y")
- price: prices, fees, rates, costs
- statistic: numbers that change over time (population, market share, user counts, rankings)
- policy: laws, regulations, guidelines, official rules and requirements
- event_status: state of an ongoing event (pending, announced, scheduled, in progress)
- relative_time: depends on "now" (latest, current, recently, this year) and fits no class above
- static: will not change (definitions, science, settled history, math)

When unsure between static and a volatile class, choose the volatile class.

Return only JSON of the form: {"items": [{"id": <claim id>, "claim_class": "<class>"}]} with one item per claim.

Claims (JSON):
{{claims}}
