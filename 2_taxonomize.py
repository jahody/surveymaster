"""Step 2: Design taxonomies with human-in-the-loop.

Phase 1: LLMs propose taxonomy axes → human selects + adds comments.
Phase 2: LLM generates full taxonomies for approved axes → human reviews.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

import httpx

import _resolvers  # noqa: F401 – registers OmegaConf resolvers
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from openai import AsyncOpenAI, RateLimitError, APITimeoutError

from dotenv import load_dotenv
load_dotenv()

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
    "You are an expert research taxonomist. Given a collection of academic papers, "
    "create multiple distinct hierarchical taxonomies that organize them in fundamentally "
    "different ways. Respond with valid JSON only."
)

REFINE_SYSTEM = (
    "You are an expert research taxonomist. Given multiple taxonomy proposals generated "
    "by different models, select the best organizing perspectives, combine the strongest "
    "categorization choices, and produce polished final taxonomies. Respond with valid JSON only."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def fetch_all_pricing(base_url: str, api_key: str) -> dict[str, tuple[float, float]]:
    """Fetch pricing for all models in one request. Returns {model_id: (prompt, completion)}."""
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            return {
                m["id"]: (float(m["pricing"].get("prompt", 0)), float(m["pricing"].get("completion", 0)))
                for m in resp.json().get("data", [])
                if "pricing" in m
            }
    except Exception as e:
        log.warning("Could not fetch model pricing: %s", e)
    return {}


def parse_json_response(text: str | None) -> list | dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    if not text:
        raise ValueError("Empty or null text received.")
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


def slugify_model(model: str) -> str:
    slug = model.split("/")[-1]
    return re.sub(r'[<>:"/\\|?*]', '_', slug)


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


def print_cost(label: str, token_usage: dict, pricing: tuple[float, float] | None) -> float:
    """Print token usage + cost for a model. Returns cost or 0."""
    total = token_usage["prompt"] + token_usage["completion"]
    if not total:
        return 0.0
    print(f"  [{label}] Tokens: {token_usage['prompt']:,} in / {token_usage['completion']:,} out / {total:,} total", end="")
    if pricing:
        cost = token_usage["prompt"] * pricing[0] + token_usage["completion"] * pricing[1]
        print(f"  ${cost:.4f}")
        return cost
    print()
    return 0.0


def load_papers(query_dir: Path, source: str) -> list[dict]:
    """Load papers from a query folder. Returns list of {id, title, content}."""
    source_dir = query_dir / source
    results_file = query_dir / "search_results.json"

    titles = {}
    if results_file.exists():
        for paper in json.loads(results_file.read_text(encoding="utf-8")):
            arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
            if arxiv_id:
                titles[arxiv_id] = paper.get("title", arxiv_id)

    papers = []
    for md_file in sorted(source_dir.glob("*.md")):
        arxiv_id = md_file.stem
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            papers.append({"id": arxiv_id, "title": titles.get(arxiv_id, arxiv_id), "content": content})
    return papers


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

def build_prompt(papers: list[dict], query_name: str) -> str:
    papers_text = ""
    for p in papers:
        papers_text += f'\n### {p["title"]} (arxiv: {p["id"]})\n{p["content"]}\n---\n'

    return f"""Here are {len(papers)} papers on the topic "{query_name}" for reference:

{papers_text}

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
    all_taxonomies: list[dict], papers: list[dict], query_name: str,
    human_comments: dict[str, str] | None = None,
) -> str:
    proposals_text = ""
    for i, entry in enumerate(all_taxonomies, 1):
        proposals_text += f'\n## Proposal {i}: "{entry["perspective"]}" from {slugify_model(entry["model"])}\n'
        if entry.get("description"):
            proposals_text += f'_{entry["description"]}_\n'
        proposals_text += f'```json\n{json.dumps(entry["categories"], indent=2)}\n```\n---\n'

    comments_text = ""
    if human_comments:
        comments_text = "\n## Human Comments (incorporate these when building the taxonomies)\n"
        for perspective, comment in human_comments.items():
            comments_text += f'- **{perspective}**: {comment}\n'
        comments_text += "\n"

    return f"""Here are {len(all_taxonomies)} taxonomy proposals for classifying papers on "{query_name}", generated by different models using different organizing perspectives:

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
# LLM calls
# ---------------------------------------------------------------------------

async def call_llm(
    client: AsyncOpenAI, system: str, user: str,
    model: str, temperature: float, max_tokens: int, token_usage: dict,
) -> list | dict:
    """Single LLM call with retry. Returns parsed JSON."""
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            usage = response.usage
            if usage:
                token_usage["prompt"] += usage.prompt_tokens
                token_usage["completion"] += usage.completion_tokens
                log.info("[%s] tokens: %d in / %d out", slugify_model(model), usage.prompt_tokens, usage.completion_tokens)
            if getattr(response, "choices", None) is None or not response.choices:
                raise ValueError("Response choices are empty or null.")
            
            return parse_json_response(response.choices[0].message.content)
        except (RateLimitError, APITimeoutError) as e:
            delay = 2 ** (attempt + 1)
            log.warning("Retry %d/3 for %s: %s (waiting %ds)", attempt + 1, model, e, delay)
            await asyncio.sleep(delay)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < 2:
                log.warning("Parse error from %s, retrying: %s", model, e)
                await asyncio.sleep(1)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for {model}")


async def generate_for_model(
    client: AsyncOpenAI, model: str, papers: list[dict],
    query_name: str, cfg: DictConfig, tax_dir: Path, query_dir: Path,
) -> dict:
    """Generate N taxonomies from one model. Returns {model, token_usage, taxonomies}."""
    slug = slugify_model(model)
    token_usage = {"prompt": 0, "completion": 0}

    print(f"  [{slug}] generating taxonomies...")
    result = await call_llm(
        client, SYSTEM_MESSAGE, build_prompt(papers, query_name),
        model, cfg.generation.temperature, cfg.generation.max_tokens, token_usage,
    )
    taxonomies = extract_taxonomies(result)

    model_dir = tax_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, tax in enumerate(taxonomies):
        save_taxonomy(tax, i, model_dir, query_dir, slug)

    return {"model": model, "token_usage": token_usage, "taxonomies": taxonomies}


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
    print("(Multiple model proposals for the same perspective will all be passed to the refiner)\n")
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

    # Save approved axes
    return filtered, human_comments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_taxonomize(cfg: DictConfig) -> None:
    query_dir = Path(to_absolute_path(cfg.generation.query_dir))
    source = cfg.generation.source
    models = list(cfg.generation.models)
    gen = cfg.generation

    query_name = query_dir.name.replace("-", " ").replace("_", " ")

    # --- Topic selection ---
    papers_base = query_dir.parent
    available = sorted(
        d for d in papers_base.iterdir()
        if d.is_dir() and any((d / source).glob("*.md"))
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

    papers = load_papers(query_dir, source)
    if not papers:
        log.error("No papers found in %s/%s", query_dir, source)
        return

    print(f"Loaded {len(papers)} papers from {query_dir / source}")
    print(f"Models: {', '.join(slugify_model(m) for m in models)}")
    print(f"Refiner: {slugify_model(gen.refiner_model)}\n")

    tax_dir = query_dir / "taxonomies"
    tax_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=gen.api_key, base_url=gen.base_url)
    total_cost = 0.0
    any_missing_pricing = False

    pricing_map = await fetch_all_pricing(gen.base_url, gen.api_key)

    # --- Phase 1: Generate proposals in parallel ---
    print("--- Generating proposals ---")
    results = await asyncio.gather(*[
        generate_for_model(client, m, papers, query_name, cfg, tax_dir, query_dir)
        for m in models
    ], return_exceptions=True)

    all_taxonomies = []
    for model, result in zip(models, results):
        slug = slugify_model(model)
        if isinstance(result, Exception):
            print(f"  [{slug}] FAILED: {result}")
            continue
        pricing = pricing_map.get(model)
        if pricing is None:
            any_missing_pricing = True
        total_cost += print_cost(slug, result["token_usage"], pricing)
        for i, tax in enumerate(result["taxonomies"]):
            perspective, categories, desc = tax_fields(tax, i)
            all_taxonomies.append({"model": model, "perspective": perspective, "description": desc, "categories": categories})

    print(f"\n{len(all_taxonomies)} proposals collected.\n")

    # --- Human gate 1: Select axes + add comments ---
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
    if len(all_taxonomies) < 2:
        print("Not enough proposals for refinement, skipping Phase 2.")
        print("Done.")
        return

    print("\n--- Phase 2: Generating full taxonomies ---")
    refiner_tokens = {"prompt": 0, "completion": 0}
    refiner = gen.refiner_model
    slug = slugify_model(refiner)
    refiner_temperature = float(getattr(gen, "refiner_temperature", 0.0))
    print(f"  [{slug}] refining {len(all_taxonomies)} proposals...")

    try:
        result = await call_llm(
            client, REFINE_SYSTEM,
            build_refine_prompt(all_taxonomies, papers, query_name, human_comments),
            refiner, refiner_temperature, gen.max_tokens, refiner_tokens,
        )
        for i, tax in enumerate(extract_taxonomies(result)):
            save_taxonomy(tax, i, tax_dir, query_dir)
        refiner_pricing = pricing_map.get(refiner)
        if refiner_pricing is None:
            any_missing_pricing = True
        total_cost += print_cost(slug, refiner_tokens, refiner_pricing)
    except Exception as e:
        print(f"  Refinement failed: {e}")
        log.error("Refinement failed: %s", e)

    # Grand total
    all_prompt = sum(r["token_usage"]["prompt"] for r in results if isinstance(r, dict)) + refiner_tokens["prompt"]
    all_completion = sum(r["token_usage"]["completion"] for r in results if isinstance(r, dict)) + refiner_tokens["completion"]
    if all_prompt + all_completion:
        print(f"\nTotal: {all_prompt:,} in / {all_completion:,} out / {all_prompt + all_completion:,} tokens", end="")
        if total_cost:
            suffix = " (partial)" if any_missing_pricing else ""
            print(f"  ${total_cost:.4f}{suffix}")
        elif any_missing_pricing:
            print("  (cost unavailable)")
        else:
            print()

    # --- Human gate 2: Review ---
    print(f"\nTaxonomies saved to: {tax_dir}")
    print(f"Review them in the Web UI at /editor or edit the JSON files directly.")
    print(f"When satisfied, run: python 3_classify.py")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(run_taxonomize(cfg))


if __name__ == "__main__":
    main()
