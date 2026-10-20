# System Prompt

You are a research paper classification expert. Classify the given paper according to the provided taxonomy. Respond with valid JSON only.

# User Prompt

## Paper Content

{paper_content}

---

## Taxonomy

Your task is to classify the paper according to a taxonomy mentioned at the end.
Select the top {top_k} most relevant leaf categories (those with no subcategories), ranked by relevance.
Assign a confidence percentage (integer, 0-100) to each. Confidences must sum to 100.

**Taxonomy: {taxonomy_name}**
**Description:** {taxonomy_description}

**Items:**
{items_text}
---

Respond with a JSON object containing:
- "filename": the paper's filename (use "{filename}")
- "chapters": an array of exactly {top_k} objects, ranked from most to least relevant, each with:
  - "chapter": the chapter number
  - "confidence": integer percentage (0-100)
  - "reason": a short reason for this classification

Example:
{{
    "filename": "{filename}",
    "chapters": [
        {{"chapter": "3.1.2(a)", "confidence": 75, "reason": "primary reason"}},
        {{"chapter": "3.2.1", "confidence": 15, "reason": "secondary reason"}},
        {{"chapter": "3.1.3", "confidence": 10, "reason": "tertiary reason"}}
    ]
}}
