# Ground-Truth Taxonomy Extraction Prompt

Use this prompt with an LLM to extract the exact taxonomy from a scientific survey paper into the project's JSON format.

---

## System Message

You are a meticulous research taxonomy extractor. Your job is to read a scientific survey paper and extract its EXACT organizational taxonomy — the hierarchical classification scheme the authors use to organize the surveyed work. You must faithfully reproduce the paper's own structure, not invent or improve upon it. Output valid JSON only.

## User Prompt

```
Below is a scientific survey paper. Extract the paper's organizational taxonomy — the hierarchical classification scheme the authors use to categorize the works they survey.

<paper>
{PAPER_CONTENT}
</paper>

## Instructions

1. **Extract, do not invent.** Reproduce the paper's own hierarchy exactly as the authors structured it. Use section numbers, headings, and subheadings from the paper. Do not add, merge, split, or rename categories.

2. **Identify which sections form the taxonomy.** Survey papers typically have:
   - Introductory/background sections (skip these)
   - A core organizational structure that classifies the surveyed approaches (EXTRACT THIS)
   - Evaluation/benchmark/future-work sections (skip these unless they form a separate classification axis)

3. **For each category node, extract:**
   - `chapter`: The section number from the paper (e.g., "3", "3.1", "3.1.2"). For sub-items at the same depth, use the paper's numbering or letter suffixes like "3.1.2(a)".
   - `name`: A snake_case identifier derived from the section heading (e.g., "Signal-Guided Control" → "signal_guided_control").
   - `description`: 2-3 sentences summarizing what the category covers, drawn from the paper's own description of that section. Use the authors' language.
   - `phrases`: 3-5 key terms or phrases the authors associate with this category (pulled from the section text).
   - `hard_negative`: A brief statement of what does NOT belong in this category, based on distinctions the paper draws between neighboring categories.

4. **For leaf categories (no sub-items), also extract:**
   - `papers`: An array of papers the authors place in this category. Each entry: `{"name": "ShortName", "arxiv": "XXXX.XXXXX"}`. Use the short name the survey uses and the arXiv ID if available. If no arXiv ID is given, use an empty string.

5. **Hierarchy rules:**
   - Non-leaf nodes have an `items` array of children and do NOT have a `papers` field.
   - Leaf nodes have a `papers` array and do NOT have an `items` field.
   - Preserve the paper's depth (typically 2-4 levels).

6. **If the paper has multiple independent classification axes** (e.g., "by method" AND "by application domain"), output each as a separate top-level object in the array.

## Output Format

A JSON array of top-level taxonomy objects:

```json
[
  {
    "chapter": "3",
    "name": "top_level_axis_name",
    "description": "What this classification axis covers.",
    "items": [
      {
        "name": "category_name",
        "chapter": "3.1",
        "description": "2-3 sentences from the paper.",
        "phrases": ["key phrase 1", "key phrase 2", "key phrase 3"],
        "hard_negative": "What does NOT belong here.",
        "items": [
          {
            "name": "leaf_category",
            "chapter": "3.1.1",
            "description": "2-3 sentences from the paper.",
            "phrases": ["phrase 1", "phrase 2"],
            "hard_negative": "What does NOT belong.",
            "papers": [
              {"name": "PaperShortName", "arxiv": "2401.12345"},
              {"name": "AnotherPaper", "arxiv": "2312.67890"}
            ]
          }
        ]
      }
    ]
  }
]
```

Now extract the taxonomy from the paper above. Output ONLY the JSON array — no commentary.
```
