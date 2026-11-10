<!-- rewrite v1 — model: JUDGE_MODEL. Placeholders: {{today}}, {{question}}, {{answer}}, {{corrections}} -->
You are editing an assistant's answer that contains out-of-date facts. Today is {{today}}.

Fix ONLY the outdated claims listed under "Corrections", using the corrected facts given there. Keep everything else in the answer as it is: wording, structure, formatting, tone and length. Do not add new facts of your own, do not remove correct content, and do not mention that the answer was edited.

After each corrected statement, add a citation marker like [1] that refers to the numbered sources below. End the answer with a "Sources:" list containing only the sources you cited, one per line, as `[n] title — url`.

Question the answer responds to:
{{question}}

Original answer:
<<<
{{answer}}
>>>

Corrections (JSON; each lists the stale claim, the corrected fact and its numbered sources):
{{corrections}}

Return only JSON of the form:
{"answer": "<the full corrected answer>"}
