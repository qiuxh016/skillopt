# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

### Rule Extraction for Names and Places
**Rule:** When the question asks for a specific name or location, extract the exact name or location provided in the context. Ensure the answer is concise and directly states the name.

### Rule for Concision
**Rule:** Keep answers brief and focused on the exact information needed. Avoid explanatory phrases unless specifically asked for an explanation.

### Guidelines for Multiple Possible Answers
If the context suggests multiple possible answers, select the shortest and most appropriate answer that directly matches the expected entity.
Always provide concise and straightforward answers. Avoid adding any extra information beyond what is necessary to correctly answer the question.

### Examples
**Context:** In 1905, she & her husband bought a farm at Hoosick Falls, N.Y., where she stayed until her death in 1961.

**Question:** Who bought a farm at Hoosick Falls?

**Answer:** <answer>Grandma Moses</answer>

**Context:** "...on my mind" appears in various song lyrics.

**Question:** Who frequently uses the phrase "...on my mind"?

**Answer:** <answer>Georgia</answer>

Ensure key terms and names are identified and used correctly to form concise and precise answers.
Verify context-relevant details accurately and succinctly before providing the answer in <answer>...</answer> tags.

Agents should focus on understanding the unique roles or achievements associated with individuals mentioned in the context.
