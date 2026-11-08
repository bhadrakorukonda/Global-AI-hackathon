<!-- extract v1 — model: SMALL_MODEL. Placeholders: {{question}}, {{text}} -->
You split an AI assistant's answer into atomic factual claims so each one can be fact-checked on its own.

Rules:
- One fact per claim. Split compound sentences ("X is A and B" -> two claims).
- Make every claim self-contained: replace pronouns and vague references with the explicit subject, using the question for context. "It costs $20" -> "ChatGPT Plus costs $20 per month".
- Keep the original wording, numbers, names, dates and qualifiers like "latest" or "current". Do not correct, update, or add facts.
- Skip opinions, advice, instructions, hedges, greetings, and content inside code.
- If there are no factual claims, return an empty list.

Return only JSON of the form: {"claims": ["...", "..."]}

Question (may be empty):
{{question}}

Answer:
{{text}}
