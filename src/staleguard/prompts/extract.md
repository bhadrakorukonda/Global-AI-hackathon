<!-- extract v4 — model: SMALL_MODEL. Placeholders: {{question}}, {{text}} -->
You split an AI assistant's answer into atomic factual claims so each one can be fact-checked on its own.

Rules:
- One fact per claim. Split compound sentences ("X is A and B" -> two claims).
- Make every claim self-contained: replace pronouns and vague references with the explicit subject, using the question for context. "It costs $20" -> "ChatGPT Plus costs $20 per month".
- Keep the original wording, numbers, names, dates and qualifiers like "latest" or "current". Do not correct, update, or add facts.
- The reader is using this answer today. If the answer presents something as the current state ("latest", "current", "now", or it answers a question about the present) but anchors it to a past point in time ("As of May 2024", "as of my last update", "currently, as of 2023"), drop that anchor and state the claim in the present tense: "As of May 2024, the latest stable Python is 3.12" -> "The latest stable version of Python is 3.12". Keep dates that are part of the fact itself: "Python 3.12 was released in October 2023" stays as it is.
- Skip opinions, advice, instructions, hedges, greetings, and content inside code.
- If there are no factual claims, return an empty list.

For each claim also write `query`: a short, neutral web search query (3–8 words) that would find the **true, up-to-date** answer on the claim's topic. Do not include the value the claim asserts, and do not add a year (one is added automatically):
- "The latest stable version of Python is 3.12." -> "latest stable Python version"
- "Sam Altman is the CEO of OpenAI." -> "OpenAI CEO"
- "GPT-4o costs $5 per million input tokens." -> "GPT-4o API price per million input tokens"
- "Python 3.12 was released in October 2023." -> "Python 3.12 release date"

For each claim also give `class`: how likely it is to go out of date. Pick exactly one:
- version: software/product version or release ("The latest Python is 3.12")
- role: who holds a position ("X is the CEO of Y")
- price: prices, fees, rates, costs
- statistic: numbers that change over time (population, market share, user counts, rankings)
- policy: laws, regulations, guidelines, official rules and requirements
- event_status: state of an ongoing event (pending, announced, scheduled, in progress)
- relative_time: depends on "now" (latest, current, recently, this year) and fits no class above
- static: will not change (definitions, science, geography, settled history, math)
When unsure between static and a volatile class, choose the volatile class.

Return only JSON of the form: {"claims": [{"claim": "...", "query": "...", "class": "..."}]}

Question (may be empty):
{{question}}

Answer:
{{text}}
