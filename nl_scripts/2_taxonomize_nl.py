"""Step 2: Design taxonomies with human-in-the-loop (NotebookLM Backend).

Phase 1: NotebookLM proposes taxonomy axes → human selects + adds comments.
Phase 2: NotebookLM generates full taxonomies for approved axes → human reviews.
"""

import asyncio
import json
import logging
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import _resolvers  # noqa: F401 – registers OmegaConf resolvers
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from notebooklm import NotebookLMClient
from utils_nl import get_or_create_notebook, sync_sources, call_notebooklm, parse_json_response

log = logging.getLogger(__name__)

TAXONOMY_SCHEMA_EXAMPLE = """\
{
  "chapter": "1",
  "name": "Top Level Category",
  "description": "2-3 sentence description of what this category covers.",
  "items": [
    {
      "name": "subcategory_name",
      "chapter": "1.1",
      "description": "2-3 sentence description.",
      "phrases": ["key phrase 1", "key phrase 2"],
      "hard_negative": "What does NOT belong here and why."
    },
    {
      "name": "another_subcategory",
      "chapter": "1.2",
      "description": "2-3 sentence description.",
      "items": [
        {
          "name": "leaf_category",
          "chapter": "1.2(a)",
          "description": "2-3 sentence description.",
          "phrases": ["phrase 1", "phrase 2"],
          "hard_negative": "What does NOT belong."
        }
      ]
    }
  ]
}"""

SYSTEM_MESSAGE = (
    "You are an expert research taxonomist. Given a collection of academic papers (provided as sources), "
    "create multiple distinct hierarchical taxonomies that organize them in fundamentally "
    "different ways. Respond with valid JSON only."
)

REFINE_SYSTEM = (
    "You are an expert research taxonomist. Given taxonomy proposals generated previously, "
    "select the best organizing perspectives, combine the strongest categorization choices, "
    "and produce polished final taxonomies. Respond with valid JSON only."
)


def extract_taxonomies(result: list | dict) -> list[dict]:
    """Normalize LLM response into a list of taxonomy dicts."""
    if isinstance(result, dict):
        return result.get("taxonomies", [])
    if isinstance(result, list):
        return result
    return []

def tax_fields(tax, i: int = 0) -> tuple[str, list, str]:
    """Extract (perspective, categories, description) from a taxonomy entry."""
    if not isinstance(tax, dict):
        return f"taxonomy_{i + 1}", tax if isinstance(tax, list) else [], ""
    return (
        tax.get("perspective", f"taxonomy_{i + 1}"),
        tax.get("categories", []),
        tax.get("description", ""),
    )

def save_taxonomy(tax, i: int, out_dir: Path, rel_dir: Path, label: str = "") -> None:
    """Save a single taxonomy to disk and print confirmation."""
    perspective, categories, desc = tax_fields(tax, i)
    out_path = out_dir / f"{perspective}.json"
    out_path.write_text(json.dumps(categories, indent=2, ensure_ascii=False), encoding="utf-8")
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}{perspective} -> {out_path.relative_to(rel_dir)}")
    if desc:
        print(f"    {desc}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_prompt(query_name: str, num_papers: int) -> str:
    return f"""The topic is "{query_name}". There are {num_papers} papers uploaded as sources in this notebook.

Create DISTINCT category taxonomies suited for classifying these papers.
Each taxonomy should use a fundamentally different organizing principle. Propose as many genuinely distinct perspectives as you see fit.

Requirements:
- Output a JSON object with a "taxonomies" key containing an array of taxonomy objects
- Each taxonomy object has: "perspective" (short name), "description" (1 sentence), and "categories" (the taxonomy array)
- Each taxonomy's "categories" is a JSON array (2-3 levels deep) following the schema below
- Each leaf category needs: name (snake_case), chapter, description (2-3 sentences), phrases (key terms), hard_negative (what does NOT belong and why)
- Chapter numbering: "1", "1.1", "1.1.1", "1.1(a)" for sub-items at the same level
- Aim for 5-15 leaf categories per taxonomy depending on paper diversity
- Use snake_case for category names
- Do NOT assign papers to categories — categories should be defined by their descriptions, phrases, and hard_negatives only

Single category schema example:
{TAXONOMY_SCHEMA_EXAMPLE}

Output format:
{{
  "taxonomies": [
    {{
      "perspective": "by_methodology",
      "description": "Organizes papers by their core technical approach.",
      "categories": [ ... ]
    }}
  ]
}}"""


def build_refine_prompt(
    all_taxonomies: list[dict], query_name: str,
    human_comments: dict[str, str] | None = None,
) -> str:
    proposals_text = ""
    for i, entry in enumerate(all_taxonomies, 1):
        model_name = entry.get("model", "NotebookLM")
        proposals_text += f'\n## Proposal {i}: "{entry["perspective"]}" from {model_name}\n'
        if entry.get("description"):
            proposals_text += f'_{entry["description"]}_\n'
        proposals_text += f'```json\n{json.dumps(entry["categories"], indent=2)}\n```\n---\n'

    comments_text = ""
    if human_comments:
        comments_text = "\n## Human Comments (incorporate these when building the taxonomies)\n"
        for perspective, comment in human_comments.items():
            comments_text += f'- **{perspective}**: {comment}\n'
        comments_text += "\n"

    return f"""Here are {len(all_taxonomies)} taxonomy proposals for classifying papers on "{query_name}":

{proposals_text}
{comments_text}
Your task: select the BEST and most useful organizing perspectives from the proposals above, then produce polished, refined versions of each. Discard redundant or overlapping perspectives — keep only those that are genuinely distinct. Keep as many or as few as warranted.

CRITICAL — the final taxonomies MUST be fundamentally different from each other:
- Each must use a genuinely distinct organizing principle
- If two proposals share a similar angle, merge them into one
- The top-level categories of each taxonomy should have minimal overlap
- A reader should be able to look at just the top-level category names and immediately see they organize the papers along completely different axes
- These taxonomies will be used to classify individual papers — prefer perspectives where the categories are mutually exclusive and clearly distinguishable

For each selected taxonomy:
- Combine the best categorization choices from any model that used a similar perspective
- Improve descriptions, phrases, and hard_negatives where needed
- Maintain 2-3 levels of hierarchy, 5-15 leaf categories
- Do NOT assign papers to categories — leave the "papers" field out entirely

Output a JSON object:
{{
  "taxonomies": [
    {{
      "perspective": "perspective_name",
      "description": "1 sentence explaining this organizing angle.",
      "categories": [ ... same schema as input ... ]
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Human-in-the-loop: Phase 1 — select axes + add comments
# ---------------------------------------------------------------------------

def prompt_axis_selection(all_taxonomies: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Show proposed perspectives and let user pick + comment.

    Returns (filtered_taxonomies, human_comments).
    """
    # Deduplicate by perspective name
    seen: dict[str, str] = {}
    for tax in all_taxonomies:
        p = tax["perspective"]
        if p not in seen:
            seen[p] = tax.get("description", "")

    perspectives = list(seen.keys())

    print("\n--- Phase 1: Select taxonomy axes ---")
    for i, p in enumerate(perspectives, 1):
        desc = seen[p]
        count = sum(1 for t in all_taxonomies if t["perspective"] == p)
        models_str = f"  ({count} proposal{'s' if count > 1 else ''})"
        print(f"  [{i:2}] {p}{models_str}")
        if desc:
            print(f"       {desc}")

    print(f"\nEnter numbers separated by commas, or 'all' [{len(perspectives)} perspectives total]:")

    selected_perspectives: set[str] = set()
    while True:
        raw = input("> ").strip()
        if raw.lower() in ("all", ""):
            selected_perspectives = set(perspectives)
            break
        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            invalid = [i for i in indices if not (1 <= i <= len(perspectives))]
            if invalid:
                print(f"  Out of range: {invalid}. Enter numbers 1-{len(perspectives)}.")
                continue
            selected_perspectives = {perspectives[i - 1] for i in indices}
            break
        except ValueError:
            print(f"  Invalid input. Use comma-separated numbers (e.g. 1,3,4) or 'all'.")

    print(f"\nSelected: {', '.join(sorted(selected_perspectives))}")

    # Collect optional comments per perspective
    human_comments: dict[str, str] = {}
    print("\nAdd optional comments/constraints per axis (press Enter to skip):")
    for p in sorted(selected_perspectives):
        comment = input(f"  {p}: ").strip()
        if comment:
            human_comments[p] = comment

    filtered = [t for t in all_taxonomies if t["perspective"] in selected_perspectives]
    print(f"\n{len(filtered)} proposals will be passed to the refiner.\n")

    return filtered, human_comments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_taxonomize(cfg: DictConfig) -> None:
    query_dir = Path(to_absolute_path(cfg.generation.query_dir))
    source = cfg.generation.source

    query_name = query_dir.name.replace("-", " ").replace("_", " ")

    # --- Topic selection ---
    papers_base = query_dir.parent
    available = sorted(
        d for d in papers_base.iterdir()
        if d.is_dir() and any((d / source).glob("*.md") or (d / source).glob("*.txt"))
    )
    if available:
        default_idx = next(
            (i for i, d in enumerate(available) if d == query_dir), None
        )
        print("Available topics:")
        for i, d in enumerate(available):
            marker = " (default)" if i == default_idx else ""
            print(f"  [{i + 1}] {d.name}{marker}")
        raw = input(f"Select topic [1-{len(available)}] (Enter = default): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            query_dir = available[int(raw) - 1]
        query_name = query_dir.name.replace("-", " ").replace("_", " ")

    papers_dir = query_dir / source
    paper_files = list(papers_dir.glob("*.md")) + list(papers_dir.glob("*.txt"))
    if not paper_files:
        log.error("No papers found in %s", papers_dir)
        return

    tax_dir = query_dir / "taxonomies"
    tax_dir.mkdir(parents=True, exist_ok=True)
    
    # Init NotebookLM Client
    print("\nInitializing NotebookLM Client (reading from ~/.notebooklm/storage_state.json)...")
    try:
        client = await NotebookLMClient.from_storage()
    except Exception as e:
        log.error(f"Failed to auth NotebookLM. Did you run 'notebooklm login'? Error: {e}")
        return
        
    async with client:
        notebook_id = await get_or_create_notebook(client, query_name)
        await sync_sources(client, notebook_id, papers_dir)

        # We will use one "model" since NotebookLM is NotebookLM. We can generate multiple proposals 
        # by calling it twice if we want, but 1 pass is usually good enough for NotebookLM.
        print("\n--- Generating proposals via NotebookLM ---")
        prompt = build_prompt(query_name, len(paper_files))
        
        all_taxonomies = []
        try:
            result = await call_notebooklm(client, notebook_id, SYSTEM_MESSAGE, prompt)
            taxonomies = extract_taxonomies(result)
            model_dir = tax_dir / "NotebookLM"
            model_dir.mkdir(parents=True, exist_ok=True)
            for i, tax in enumerate(taxonomies):
                save_taxonomy(tax, i, model_dir, query_dir, "NotebookLM")
                perspective, categories, desc = tax_fields(tax, i)
                all_taxonomies.append({"model": "NotebookLM", "perspective": perspective, "description": desc, "categories": categories})
        except Exception as e:
            print(f"  [NotebookLM] FAILED: {e}")

        print(f"\n{len(all_taxonomies)} proposals collected.\n")

        # --- Human gate 1: Select axes + add comments ---
        if not all_taxonomies:
             print("No taxonomy proposals were generated. Exiting.")
             return
             
        all_taxonomies, human_comments = prompt_axis_selection(all_taxonomies)
        if not all_taxonomies:
            print("No perspectives selected. Exiting.")
            return

        # Save approved axes
        axes_path = query_dir / "axes_approved.json"
        axes_data = [
            {"perspective": t["perspective"], "description": t.get("description", ""),
             "comment": human_comments.get(t["perspective"], "")}
            for t in all_taxonomies
        ]
        # Deduplicate
        seen_axes = set()
        unique_axes = []
        for a in axes_data:
            if a["perspective"] not in seen_axes:
                seen_axes.add(a["perspective"])
                unique_axes.append(a)
        axes_path.write_text(json.dumps(unique_axes, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Approved axes saved to: {axes_path}")

        # --- Phase 2: Refine ---
        if len(all_taxonomies) < 1:
            print("No proposals for refinement, skipping Phase 2.")
            print("Done.")
            return

        print("\n--- Phase 2: Generating full taxonomies (NotebookLM) ---")
        print(f"  [NotebookLM] refining proposals...")

        refine_prompt = build_refine_prompt(all_taxonomies, query_name, human_comments)
        try:
            result = await call_notebooklm(client, notebook_id, REFINE_SYSTEM, refine_prompt)
            for i, tax in enumerate(extract_taxonomies(result)):
                save_taxonomy(tax, i, tax_dir, query_dir)
        except Exception as e:
            print(f"  Refinement failed: {e}")
            log.error("Refinement failed: %s", e)

        # --- Human gate 2: Review ---
        print(f"\nTaxonomies saved to: {tax_dir}")
        print(f"Review them in the Web UI at /editor or edit the JSON files directly.")
        print(f"When satisfied, run: python 3_classify_nl.py")

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(run_taxonomize(cfg))


if __name__ == "__main__":
    main()
