"""Generate an evolution tree showing how papers build upon each other over time.

Uses multi-model LLM consensus: Phase 1 generates proposals in parallel,
Phase 2 refines them into a single coherent evolution tree.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
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

SYSTEM_MESSAGE = (
    "You are an expert research historian specializing in tracking the evolution "
    "of ideas in academic research. Given a collection of papers with their content "
    "and taxonomy classification, identify how papers build upon, extend, or are "
    "influenced by earlier work. Respond with valid JSON only."
)

REFINE_SYSTEM = (
    "You are an expert research historian. Given multiple evolution tree proposals "
    "from different models, merge them into a single coherent evolution tree. "
    "Resolve conflicts, remove duplicate or weak relationships, and ensure the "
    "final tree accurately represents the research lineage. Respond with valid JSON only."
)


# ---------------------------------------------------------------------------
# Helpers (shared with 3_generate_taxonomy.py)
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


def parse_json_response(text: str) -> list | dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


def slugify_model(model: str) -> str:
    return model.split("/")[-1]


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
    print("  (cost unavailable)")
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
            return parse_json_response(response.choices[0].message.content)
        except (RateLimitError, APITimeoutError) as e:
            delay = 2 ** (attempt + 1)
            log.warning("Retry %d/3 for %s: %s (waiting %ds)", attempt + 1, model, e, delay)
            await asyncio.sleep(delay)
        except json.JSONDecodeError as e:
            if attempt < 2:
                log.warning("JSON parse error from %s, retrying: %s", model, e)
                await asyncio.sleep(1)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for {model}")


# ---------------------------------------------------------------------------
# Evolution-specific helpers
# ---------------------------------------------------------------------------

def parse_arxiv_date(arxiv_id: str) -> tuple[int, int] | None:
    """Extract (year, month) from arxiv ID prefix. E.g. '2501.12948' -> (2025, 1)."""
    m = re.match(r"^(\d{2})(\d{2})\.\d+", arxiv_id)
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    year = 2000 + yy if yy < 90 else 1900 + yy
    return (year, mm)


def assign_time_period(year: int, month: int) -> str:
    """Bucket a paper into a time period for the evolution tree layout."""
    if year <= 2022:
        return f"Y{year}"
    if year == 2023:
        return "Y2023"
    if year == 2024:
        return f"Y2024_H{'1' if month <= 6 else '2'}"
    # 2025+
    quarter = (month - 1) // 3 + 1
    return f"Y{year}_Q{quarter}"


def load_taxonomy_lookups(taxonomies_dir: Path) -> dict[str, dict[str, dict]]:
    """Load all taxonomy JSON files into a chapter lookup table."""
    lookups = {}
    
    def walk(items, path_parts, current_lookup):
        for item in items:
            current = path_parts + [item["name"]]
            if "chapter" in item:
                current_lookup[item["chapter"]] = {
                    "name": item["name"],
                    "full_path": " / ".join(current)
                }
            if "items" in item:
                walk(item["items"], current, current_lookup)

    for path in sorted(taxonomies_dir.glob("*.json")):
        taxonomy_name = path.stem
        taxonomy_data = json.loads(path.read_text(encoding="utf-8"))
        
        current_lookup = {}
        walk(taxonomy_data, [], current_lookup)
        lookups[taxonomy_name] = current_lookup
        log.info("Loaded taxonomy lookup %s", taxonomy_name)
        
    return lookups


def build_category_map(consensus_dir: Path, lookups: dict) -> dict[str, list[dict]]:
    """Map arxiv_id -> list of {taxonomy, chapter, name, full_path} from consensus classifications."""
    cat_map: dict[str, list[dict]] = {}
    files = sorted(consensus_dir.glob("*.json"))
    if not files:
        log.warning("No consensus files found in %s", consensus_dir)
        return cat_map
        
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        arxiv_id = path.stem
        classifications = data.get("classifications", {})
        
        for tax_name, tax_result in classifications.items():
            chapter = tax_result.get("chapter")
            if chapter and tax_name in lookups and chapter in lookups[tax_name]:
                lookup_info = lookups[tax_name][chapter]
                cat_map.setdefault(arxiv_id, []).append({
                    "taxonomy": tax_name,
                    "chapter": chapter,
                    "name": lookup_info["name"],
                    "full_path": lookup_info["full_path"],
                })
    return cat_map



def create_chronological_chunks(
    papers: list[dict], chunk_size: int, overlap: int,
) -> list[list[dict]]:
    """Split papers into overlapping chronological windows for better LLM attention."""
    # Sort chronologically
    dated = [(parse_arxiv_date(p["id"]) or (9999, 99), p) for p in papers]
    dated.sort(key=lambda x: x[0])
    sorted_papers = [p for _, p in dated]

    if len(sorted_papers) <= chunk_size:
        return [sorted_papers]

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(sorted_papers), step):
        chunk = sorted_papers[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(sorted_papers):
            break
    return chunks


def _prompt_topic_selection(default_query_dir: Path, source: str) -> Path:
    """Let user pick a topic dir at runtime. Returns the chosen topic dir."""
    papers_base = default_query_dir.parent
    available = sorted(
        d for d in papers_base.iterdir()
        if d.is_dir() and any((d / source).glob("*.md"))
    )
    if not available:
        return default_query_dir
    default_idx = next((i for i, d in enumerate(available) if d == default_query_dir), None)
    print("Available topics:")
    for i, d in enumerate(available):
        marker = " (default)" if i == default_idx else ""
        print(f"  [{i + 1}] {d.name}{marker}")
    raw = input(f"Select topic [1-{len(available)}] (Enter = default): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(available):
        return available[int(raw) - 1]
    return default_query_dir


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_evolution_prompt(
    papers: list[dict], category_map: dict, query_name: str,
    content_chars: int = 1200,
    chunk_index: int | None = None, total_chunks: int | None = None,
) -> str:
    # Sort papers chronologically
    dated = []
    for p in papers:
        d = parse_arxiv_date(p["id"])
        dated.append((d or (9999, 99), p))
    dated.sort(key=lambda x: x[0])

    papers_text = ""
    for (year, month), p in dated:
        cats = category_map.get(p["id"], [])
        if cats:
            parts = [f'{c["taxonomy"]}: {c["chapter"]} {c["name"]}' for c in cats]
            cat_label = f' [{"; ".join(parts)}]'
        else:
            cat_label = ""
        date_label = f"{year}-{month:02d}" if year < 9999 else "unknown"
        content = p["content"][:content_chars] + "..." if len(p["content"]) > content_chars else p["content"]
        papers_text += f'\n### {p["title"]} (arxiv: {p["id"]}, {date_label}){cat_label}\n{content}\n---\n'

    chunk_context = ""
    if chunk_index is not None and total_chunks is not None:
        chunk_context = f"\nNote: This is chunk {chunk_index + 1} of {total_chunks}. Focus on relationships within this group of papers.\n"

    return f"""Here are {len(papers)} papers on "{query_name}", listed chronologically:

{papers_text}
{chunk_context}
Your task: Construct an evolution tree showing how these research ideas build on each other over time. Be thorough — find as many meaningful relationships as possible.

For each relationship, provide:
- source: arxiv ID of the earlier paper
- target: arxiv ID of the later paper that builds upon it
- type: one of the four types below
- label: short phrase (2-5 words) describing the specific connection

Edge types:
- "direct_extension" — Paper B explicitly extends, improves, or reimplements Paper A's method/architecture. Look for: same model name with improvements, v2/v3 versions, papers that modify a specific algorithm from Paper A.
- "conceptual_influence" — Paper B adapts or is clearly inspired by a key idea from Paper A, even if the implementation differs. This is the most common type — use it generously for any clear intellectual lineage.
- "shared_benchmark" — Both papers evaluate on the same specific benchmark (e.g., MMLU, GSM8K, HumanEval, MATH, MT-Bench). Include this even alongside other edge types.
- "challenges" — Paper B explicitly argues against, refutes, or demonstrates fundamental limitations of Paper A's approach.

Guidelines:
- IMPORTANT: Every paper should appear in at least one edge (as source or target). If a paper seems isolated, look harder for conceptual connections to the most related paper.
- When in doubt about whether a relationship exists, include it as "conceptual_influence" — it is better to include a plausible relationship than to miss a real one.
- A paper can have multiple predecessors and successors — aim for 1-3 incoming edges per paper.
- Papers from the same time period working on similar sub-problems likely share influences from the same earlier work.
- Edges must flow from earlier to later papers (check the arxiv ID dates: YYMM format).
- Use "direct_extension" for clear technical building blocks (not just topical similarity).

Output JSON:
{{
  "edges": [
    {{
      "source": "2201.11903",
      "target": "2501.12948",
      "type": "direct_extension",
      "label": "pure RL reasoning"
    }}
  ]
}}"""


def build_refine_prompt(
    all_proposals: list[dict], papers: list[dict], query_name: str,
    chunk_mode: bool = False,
) -> str:
    proposals_text = ""
    for i, entry in enumerate(all_proposals, 1):
        slug = slugify_model(entry["model"])
        proposals_text += f'\n## Proposal {i} from {slug} ({len(entry["edges"])} edges)\n'
        proposals_text += f'```json\n{json.dumps(entry["edges"], indent=2)}\n```\n---\n'

    paper_list = ""
    for p in papers:
        d = parse_arxiv_date(p["id"])
        ds = f"{d[0]}-{d[1]:02d}" if d else "unknown"
        paper_list += f'- {p["title"]} (arxiv: {p["id"]}, {ds})\n'

    chunk_note = ""
    if chunk_mode:
        chunk_note = (
            "\nNote: Proposals were generated from overlapping paper chunks, so many edges "
            "appear in only one proposal. This is expected — do NOT remove edges just because "
            "they come from a single source.\n"
        )

    return f"""Here are {len(all_proposals)} evolution tree proposals for {len(papers)} papers on "{query_name}":

{proposals_text}

Papers:
{paper_list}
{chunk_note}
Merge these proposals into ONE coherent, DENSE evolution tree.

Target density: approximately {len(papers)} edges for {len(papers)} papers (roughly 1-2 edges per paper). The goal is a well-connected tree where most papers participate.

Edge types: "direct_extension", "conceptual_influence", "shared_benchmark", "challenges"

Rules:
1. **Consensus**: edges proposed by multiple models are high-confidence — always keep them
2. **Conflicts**: if models disagree on type/label for the same source->target, pick the most accurate
3. **Single-source edges**: keep them unless they are clearly wrong (nonsensical relationship or wrong chronological direction). Do NOT bulk-remove edges just because only one source proposed them.
4. **Deduplicate**: if multiple edges exist between the same two papers, keep only the strongest/most specific
5. **Chronology**: edges must flow from earlier to later papers (check arxiv ID dates: YYMM prefix)
6. **Coverage check**: after merging, verify that every paper appears in at least one edge. For any isolated paper, add a "conceptual_influence" edge connecting it to the most topically related paper.

Output the final merged tree:
{{
  "edges": [
    {{
      "source": "arxiv_id",
      "target": "arxiv_id",
      "type": "direct_extension",
      "label": "short phrase"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

async def generate_for_model(
    client: AsyncOpenAI, model: str, papers: list[dict],
    category_map: dict, query_name: str, cfg: DictConfig, out_dir: Path,
    chunks: list[list[dict]] | None = None,
) -> dict:
    """Generate evolution edges from one model. Returns {model, token_usage, edges}."""
    slug = slugify_model(model)
    token_usage = {"prompt": 0, "completion": 0}
    gen = cfg.evolution
    content_chars = getattr(gen, "content_chars", 1200)

    if chunks and len(chunks) > 1:
        # Process each chunk sequentially, collect all edges
        all_edges = []
        for i, chunk in enumerate(chunks):
            print(f"  [{slug}] chunk {i + 1}/{len(chunks)} ({len(chunk)} papers)...")
            result = await call_llm(
                client, SYSTEM_MESSAGE,
                build_evolution_prompt(
                    chunk, category_map, query_name,
                    content_chars=content_chars,
                    chunk_index=i, total_chunks=len(chunks),
                ),
                model, gen.temperature, gen.max_tokens, token_usage,
            )
            chunk_edges = result.get("edges", []) if isinstance(result, dict) else []
            all_edges.extend(chunk_edges)
            print(f"    chunk {i + 1}: {len(chunk_edges)} edges")

        # Deduplicate by (source, target), keep first occurrence
        seen = set()
        edges = []
        for e in all_edges:
            key = (e.get("source"), e.get("target"))
            if key not in seen:
                seen.add(key)
                edges.append(e)
        print(f"  [{slug}] {len(all_edges)} raw -> {len(edges)} deduplicated edges")
    else:
        print(f"  [{slug}] proposing evolution relationships...")
        result = await call_llm(
            client, SYSTEM_MESSAGE,
            build_evolution_prompt(papers, category_map, query_name, content_chars=content_chars),
            model, gen.temperature, gen.max_tokens, token_usage,
        )
        edges = result.get("edges", []) if isinstance(result, dict) else []

    # Save individual proposal
    proposals_dir = out_dir / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    out_path = proposals_dir / f"{slug}.json"
    out_path.write_text(json.dumps(edges, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    {len(edges)} edges -> proposals/{slug}.json")

    return {"model": model, "token_usage": token_usage, "edges": edges}


def validate_and_patch_edges(edges: list[dict], papers: list[dict]) -> list[dict]:
    """Validate edges: remove invalid, deduplicate, log coverage."""
    valid_ids = {p["id"] for p in papers}
    dates = {p["id"]: parse_arxiv_date(p["id"]) for p in papers}

    seen = set()
    cleaned = []
    removed_unknown = 0
    removed_chrono = 0
    removed_dup = 0

    for e in edges:
        src, tgt = e.get("source"), e.get("target")

        # Skip edges referencing unknown papers
        if src not in valid_ids or tgt not in valid_ids:
            removed_unknown += 1
            continue

        # Skip wrong chronological direction
        src_date, tgt_date = dates.get(src), dates.get(tgt)
        if src_date and tgt_date and src_date > tgt_date:
            removed_chrono += 1
            continue

        # Deduplicate
        key = (src, tgt)
        if key in seen:
            removed_dup += 1
            continue
        seen.add(key)
        cleaned.append(e)

    # Coverage stats
    covered = set()
    for e in cleaned:
        covered.add(e["source"])
        covered.add(e["target"])
    uncovered = valid_ids - covered

    if removed_unknown or removed_chrono or removed_dup:
        print(f"  Validation: removed {removed_unknown} unknown-ID, {removed_chrono} wrong-direction, {removed_dup} duplicate edges")
    print(f"  Coverage: {len(covered)}/{len(valid_ids)} papers have edges", end="")
    if uncovered:
        print(f" ({len(uncovered)} isolated: {', '.join(sorted(uncovered)[:5])}{'...' if len(uncovered) > 5 else ''})")
    else:
        print(" (full coverage)")

    return cleaned


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_final_output(
    edges: list[dict], papers: list[dict], query_name: str,
) -> dict:
    """Assemble the full evolution_tree.json structure (taxonomy-agnostic)."""
    nodes = []
    for p in papers:
        d = parse_arxiv_date(p["id"])
        year, month = d if d else (None, None)
        tp = assign_time_period(year, month) if d else "Unknown"

        nodes.append({
            "arxiv": p["id"],
            "name": p["title"],
            "time_period": tp,
            "year": year,
            "month": month,
        })

    nodes.sort(key=lambda n: (n["year"] or 9999, n["month"] or 99))

    # Collect unique time periods in order
    seen = set()
    time_periods = []
    for n in nodes:
        tp = n["time_period"]
        if tp != "Unknown" and tp not in seen:
            seen.add(tp)
            time_periods.append({"id": tp, "label": tp.replace("_", " ").lstrip("Y")})

    return {
        "metadata": {
            "query": query_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_papers": len(nodes),
            "total_edges": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
        "time_periods": time_periods,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_evolution(cfg: DictConfig) -> None:
    gen = cfg.evolution
    source = gen.source
    query_dir = _prompt_topic_selection(Path(to_absolute_path(gen.query_dir)), source)
    models = list(gen.models)

    query_name = query_dir.name.replace("-", " ").replace("_", " ")
    papers = load_papers(query_dir, source)
    if not papers:
        log.error("No papers found in %s/%s", query_dir, source)
        return

    taxonomies_dir = query_dir / "taxonomies"
    consensus_dir = query_dir / "classification" / "_consensus"
    
    if not taxonomies_dir.exists():
        log.error("Taxonomies directory not found: %s", taxonomies_dir)
        return
    if not consensus_dir.exists():
        log.error("Consensus directory not found: %s", consensus_dir)
        return

    taxonomy_lookups = load_taxonomy_lookups(taxonomies_dir)
    if not taxonomy_lookups:
        log.error("No taxonomy files loaded from %s", taxonomies_dir)
        return
        
    category_map = build_category_map(consensus_dir, taxonomy_lookups)

    # Chunking config
    chunk_size = getattr(gen, "chunk_size", 20)
    chunk_overlap = getattr(gen, "chunk_overlap", 5)

    if chunk_size > 0 and len(papers) > chunk_size:
        chunks = create_chronological_chunks(papers, chunk_size, chunk_overlap)
    else:
        chunks = None

    print(f"Loaded {len(papers)} papers from {query_dir / source}")
    if chunks:
        print(f"Chunking: {len(chunks)} chunks of ~{chunk_size} papers (overlap={chunk_overlap})")
    print(f"Taxonomies loaded: {', '.join(taxonomy_lookups.keys())}")
    print(f"Classifications mapped: {len(category_map)} papers")
    print(f"Models: {', '.join(slugify_model(m) for m in models)}")
    print(f"Refiner: {slugify_model(gen.refiner_model)}\n")

    out_dir = query_dir / "evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=gen.api_key, base_url=gen.base_url)
    total_cost = 0.0
    any_missing_pricing = False
    pricing_map = await fetch_all_pricing(gen.base_url, gen.api_key)

    # --- Phase 1: Generate proposals in parallel ---
    print("--- Phase 1: Generating proposals ---")
    results = await asyncio.gather(*[
        generate_for_model(client, m, papers, category_map, query_name, cfg, out_dir, chunks=chunks)
        for m in models
    ], return_exceptions=True)

    all_proposals = []
    for model, result in zip(models, results):
        slug = slugify_model(model)
        if isinstance(result, Exception):
            print(f"  [{slug}] FAILED: {result}")
            continue
        pricing = pricing_map.get(model)
        if pricing is None:
            any_missing_pricing = True
        total_cost += print_cost(slug, result["token_usage"], pricing)
        all_proposals.append({"model": model, "edges": result["edges"]})

    print(f"\n{len(all_proposals)} proposals collected.\n")

    # --- Phase 2: Refine ---
    if len(all_proposals) < 2:
        print("Not enough proposals for refinement, skipping Phase 2.")
        print("Done.")
        return

    print("--- Phase 2: Refining ---")
    refiner_tokens = {"prompt": 0, "completion": 0}
    refiner = gen.refiner_model
    slug = slugify_model(refiner)
    print(f"  [{slug}] merging {len(all_proposals)} proposals...")

    try:
        result = await call_llm(
            client, REFINE_SYSTEM,
            build_refine_prompt(all_proposals, papers, query_name, chunk_mode=chunks is not None),
            refiner, gen.temperature, gen.max_tokens, refiner_tokens,
        )
        refined_edges = result.get("edges", []) if isinstance(result, dict) else []

        # Validate and clean edges
        refined_edges = validate_and_patch_edges(refined_edges, papers)

        # Build and save final output
        final = build_final_output(refined_edges, papers, query_name)

        output_path = out_dir / "evolution_tree.json"
        output_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {output_path.relative_to(query_dir)}")
        print(f"  {final['metadata']['total_papers']} papers, {final['metadata']['total_edges']} edges")

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
    print("Done.")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(run_evolution(cfg))


if __name__ == "__main__":
    main()
