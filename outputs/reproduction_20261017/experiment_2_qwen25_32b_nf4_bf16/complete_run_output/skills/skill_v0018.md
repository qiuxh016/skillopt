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

### Final Answer Verification
Before finalizing the answer, double-check that it conforms strictly to the context and avoids any superfluous information. Ensure the minimal necessary details are provided without deviation.

### Direct Concise Answers<br/>**Rule:** Your answer should directly match the required entity (name, location, etc.) without any supplementary information or qualifications.<br/>

**Rule:** Always verify the entity extracted from the context is the one anticipated by the question.

### Ensuring Precise Responses<br/>**Rule:** Follow the context's exact wording when forming answers, adhering strictly to required names, locations, titles, or extra details as specified (e.g., quote marks, honorifics).

### Entity Matching Verification<br/>**Rule:** Cross-reference the extracted names and locations with the question to ensure accuracy and specificity before submitting the answer.

<br/><br/>### Reinforce Verification Against Questions
**Rule:** Always cross-reference the extracted entity (names, locations) with the exact language and context of the question to ensure accurate and contextually relevant answers.

### Confirm Conciseness
**Rule:** Ensure that the final answer avoids extraneous information and qualifies the answer only as directly implied by the context and question.

### Specific Entity Extraction<br>**Rule:** Always extract the exact entity name as specified in the context without adding any extra descriptors.

### Clarification of Specificity over Detail<br/>**Rule:** Prioritize matching context and question specificity over providing extensive background or detail which leads to incorrect verbosity.

### Rule for Selecting Shortest Answer<br>**Rule:** If the context provides multiple synonyms or descriptions of the same entity, choose the shortest and most commonly used name.

<br/>
### Prefer Shortest Name When Exact Matching<br/>
**Rule:** When the context and question specify a shorter, exact name, always prioritize matching and answering with the shortest entity name. Avoid including any qualifiers, additional information, or longer names that deviate from the required name specificity.<br/>

<br/>
### Avoid Qualifiers When Extracting Entities<br/>
**Rule:** When extracting a name or location, refrain from adding any additional qualifiers, descriptions, or modifiers to the entity. Give the answer in the form and format specified by the context or question, minimizing any ancillary information.<br/>

<br/>### Prevent Unnecessary Elaborations
**Rule:** Ensure that answers are strictly factual and avoid elaborating beyond the explicit requirements of the question. The extracted information should directly correspond to the context and question without additional context.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
