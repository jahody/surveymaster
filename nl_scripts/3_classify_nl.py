"""Step 3: Classify papers against taxonomies (NotebookLM Backend)."""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import _resolvers  # noqa: F401 – registers OmegaConf resolvers
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from notebooklm import NotebookLMClient
from utils_nl import get_or_create_notebook, sync_sources, call_notebooklm

log = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "You are a research paper classification expert. "
    "Classify the paper in your sources according to the provided taxonomy. "
    "Respond with valid JSON only."
)

def collect_leaf_items(items: list[dict]) -> list[dict]:
    """Recursively collect only leaf items (those with no sub-items)."""
    leaves = []
    for item in items:
        children = item.get("items", [])
        if children:
            leaves.extend(collect_leaf_items(children))
        else:
            leaves.append(item)
    return leaves


def build_prompt(taxonomy: dict, filename: str, top_k: int = 3) -> str:
    """Build the user message for classifying one paper (as source) against one taxonomy."""
    leaves = collect_leaf_items(taxonomy["items"])
    items_text = ""
    for item in leaves:
        name = item.get("name", "Unknown")
        chapter = str(item.get("chapter", name))  # fall back to item name as identifier
        items_text += f'\n### {name} (chapter: {chapter})\n'
        items_text += f'**Description:** {item.get("description", "")}\n'
        items_text += f'**Key phrases:** {", ".join(item.get("phrases", []))}\n'
        items_text += f'**Hard negative (DO NOT assign if this describes the paper\'s approach):** {item.get("hard_negative", "")}\n'

    return f"""## Taxonomy

Your task is to classify the attached source paper according to the taxonomy below.
Select the top {top_k} most relevant leaf categories (those with no subcategories), ranked by relevance.
Assign a confidence percentage (integer, 0-100) to each. Confidences must sum to 100.

**Taxonomy: {taxonomy["name"]}**
**Description:** {taxonomy["description"]}

**Items:**
{items_text}
---

Respond with a JSON object containing:
- "filename": the paper's filename (use "{filename}")
- "chapters": an array of exactly {top_k} objects, ranked from most to least relevant, each with:
  - "chapter": the chapter identifier exactly as shown above (e.g. "1.2" or "Model Switch")
  - "confidence": integer percentage (0-100)
  - "reason": a short reason for this classification

Example:
{{
    "filename": "{filename}",
    "chapters": [
        {{"chapter": "3.1.2", "confidence": 75, "reason": "primary reason"}},
        {{"chapter": "Model Switch", "confidence": 15, "reason": "secondary reason"}},
        {{"chapter": "3.1.3", "confidence": 10, "reason": "tertiary reason"}}
    ]
}}"""


async def classify_one(
    client: NotebookLMClient,
    notebook_id: str,
    source_id: str,
    taxonomy: dict,
    filename: str,
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Classify a single paper against a single taxonomy using NotebookLM."""
    prompt = build_prompt(taxonomy, filename, top_k)
    log.debug("Prompt length for '%s': %d chars", taxonomy["name"], len(prompt))

    async with semaphore:
        try:
            return await call_notebooklm(
                client=client,
                notebook_id=notebook_id,
                system_prompt=SYSTEM_MESSAGE,
                user_prompt=prompt,
                source_ids=[source_id]
            )
        except Exception as e:
            log.error("Error for taxonomy '%s': %s", taxonomy["name"], e)
            return {"error": str(e)}


async def classify_paper(
    client: NotebookLMClient,
    notebook_id: str,
    source_id: str,
    paper_path: Path,
    taxonomies: list[dict],
    top_k: int,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    skip_existing: bool,
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
) -> dict | None:
    """Classify one paper against all taxonomies and write result JSON."""
    output_path = output_dir / f"{paper_path.stem}.json"

    existing_classifications = {}
    if output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text(encoding="utf-8"))
            existing_classifications = existing_data.get("classifications", {})
        except Exception:
            pass

    taxonomies_to_run = taxonomies
    if skip_existing:
        taxonomies_to_run = [
            tax for tax in taxonomies 
            if tax["name"] not in existing_classifications or "error" in existing_classifications[tax["name"]]
        ]

    if not taxonomies_to_run:
        log.info("SKIP %s (all taxonomies exist)", paper_path.name)
        return None

    log.info("Classifying %s (running %d/%d taxonomies) ...", paper_path.name, len(taxonomies_to_run), len(taxonomies))

    tasks = [
        classify_one(
            client=client, notebook_id=notebook_id, source_id=source_id,
            taxonomy=tax, filename=paper_path.name,
            top_k=top_k, semaphore=semaphore,
        )
        for tax in taxonomies_to_run
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    classifications = existing_classifications.copy()
    for taxonomy, result in zip(taxonomies_to_run, results):
        if isinstance(result, Exception):
             classifications[taxonomy["name"]] = {"error": str(result)}
        else:
             classifications[taxonomy["name"]] = result

    if taxonomy_lookup:
        _normalize_result(classifications, taxonomy_lookup)

    output = {
        "filename": paper_path.name,
        "classifications": classifications,
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("OK %s", paper_path.name)
    return output


def _build_name_to_chapter(items: list[dict], mapping: dict) -> None:
    for item in items:
        if "name" in item and "chapter" in item:
            mapping[item["name"]] = str(item["chapter"])
        if "items" in item:
            _build_name_to_chapter(item["items"], mapping)

def _load_taxonomy_lookup(taxonomies_dir: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for json_path in sorted(taxonomies_dir.glob("*.json")):
        mapping: dict[str, str] = {}
        items = json.loads(json_path.read_text(encoding="utf-8"))
        _build_name_to_chapter(items, mapping)
        lookup[json_path.stem] = mapping
    return lookup

def _normalize_chapter(chapter: str, tax_name: str, taxonomy_lookup: dict[str, dict[str, str]]) -> str:
    name_map = taxonomy_lookup.get(tax_name, {})
    return name_map.get(chapter, chapter)

def _normalize_result(classifications: dict, taxonomy_lookup: dict[str, dict[str, str]]) -> None:
    for tax_name, tax_result in classifications.items():
        if "error" in tax_result:
            continue
        name_map = taxonomy_lookup.get(tax_name, {})
        for entry in tax_result.get("chapters", []):
            if isinstance(entry, dict) and "chapter" in entry:
                entry["chapter"] = name_map.get(str(entry["chapter"]), str(entry["chapter"]))

def _find_model_dirs(output_dir: Path) -> list[Path]:
    model_dirs = []
    for json_file in output_dir.rglob("*.json"):
        model_dir = json_file.parent
        if "_consensus" not in model_dir.parts and model_dir not in model_dirs:
            model_dirs.append(model_dir)
    return sorted(model_dirs)

def _load_classifications(model_dir: Path) -> dict[str, dict]:
    results = {}
    for json_path in sorted(model_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        results[data["filename"]] = data
    return results

def aggregate_votes(
    all_model_results: dict[str, dict[str, dict]],
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict]:
    """Aggregate classifications across models using confidence-weighted voting."""
    all_papers = set()
    for model_results in all_model_results.values():
        all_papers.update(model_results.keys())

    consensus = {}
    for paper_filename in sorted(all_papers):
        paper_taxonomies = defaultdict(lambda: defaultdict(float))
        model_count = 0

        for model_name, model_results in all_model_results.items():
            paper_data = model_results.get(paper_filename)
            if not paper_data:
                continue
            model_count += 1

            for tax_name, tax_result in paper_data.get("classifications", {}).items():
                if "error" in tax_result:
                    continue
                chapters = tax_result.get("chapters", [])
                if not chapters and "chapter" in tax_result:
                    ch = tax_result["chapter"]
                    if isinstance(ch, list):
                        conf_each = 100 // max(len(ch), 1)
                        chapters = [{"chapter": c, "confidence": conf_each} for c in ch]
                    else:
                        chapters = [{"chapter": ch, "confidence": 100}]

                for entry in chapters:
                    if isinstance(entry, dict):
                        chapter = str(entry.get("chapter", "unknown"))
                        try:
                            confidence = float(entry.get("confidence", 100))
                        except (ValueError, TypeError):
                            confidence = 100.0 / max(len(chapters), 1)
                    else:
                        chapter = str(entry)
                        confidence = 100.0 / max(len(chapters), 1)
                    if taxonomy_lookup:
                        chapter = _normalize_chapter(chapter, tax_name, taxonomy_lookup)
                    paper_taxonomies[tax_name][chapter] += confidence

        classifications = {}
        for tax_name, chapter_scores in paper_taxonomies.items():
            sorted_chapters = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)
            winner_chapter, winner_score = sorted_chapters[0]

            top1_count = 0
            for model_results in all_model_results.values():
                paper_data = model_results.get(paper_filename)
                if not paper_data:
                    continue
                tax_result = paper_data.get("classifications", {}).get(tax_name, {})
                chapters = tax_result.get("chapters", [])
                if not chapters and "chapter" in tax_result:
                    ch = tax_result["chapter"]
                    if isinstance(ch, list):
                        chapters = [{"chapter": c, "confidence": 100 // max(len(ch), 1)} for c in ch]
                    else:
                        chapters = [{"chapter": ch, "confidence": 100}]
                first = chapters[0] if chapters else None
                first_ch = str(first["chapter"]) if isinstance(first, dict) else str(first) if first else None
                if first_ch and taxonomy_lookup:
                    first_ch = _normalize_chapter(first_ch, tax_name, taxonomy_lookup)
                if first_ch == winner_chapter:
                    top1_count += 1

            classifications[tax_name] = {
                "chapter": winner_chapter,
                "score": winner_score,
                "votes": {ch: int(sc) for ch, sc in sorted_chapters},
                "reason": f"Consensus: {top1_count}/{model_count} models ranked this #1",
            }

        consensus[paper_filename] = {
            "filename": paper_filename,
            "classifications": classifications,
        }

    return consensus

def _print_aggregate_summary(
    consensus: dict[str, dict],
    all_model_results: dict[str, dict[str, dict]],
) -> None:
    model_count = len(all_model_results)
    unanimous = 0
    majority = 0
    contested = 0

    for paper_filename, paper_data in sorted(consensus.items()):
        for tax_name, tax_result in paper_data["classifications"].items():
            reason = tax_result["reason"]
            top1 = int(reason.split("/")[0].split()[-1])

            if top1 == model_count:
                unanimous += 1
            elif top1 > model_count / 2:
                majority += 1
            else:
                contested += 1
                log.info(
                    "CONTESTED %s [%s]: winner=%s scores=%s",
                    paper_filename, tax_name,
                    tax_result["chapter"], tax_result["votes"],
                )

    total = unanimous + majority + contested
    print(f"\nAggregation complete ({model_count} models, {len(consensus)} papers)")
    print(f"  Unanimous:  {unanimous}/{total}")
    print(f"  Majority:   {majority}/{total}")
    print(f"  Contested:  {contested}/{total}")

def _run_aggregation_from_dirs(source_dir: Path, taxonomies_dir: Path) -> None:
    consensus_dir = source_dir / "_consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = _find_model_dirs(source_dir)
    model_dirs = [d for d in model_dirs if "_consensus" not in d.parts]

    if not model_dirs:
        log.error("No model directories found under %s", source_dir)
        return

    print(f"\n--- Aggregation ---")
    print(f"Found {len(model_dirs)} model directories:")
    for d in model_dirs:
        print(f"  {d.relative_to(source_dir)}")

    all_model_results = {}
    for model_dir in model_dirs:
        model_name = str(model_dir.relative_to(source_dir))
        all_model_results[model_name] = _load_classifications(model_dir)
        log.info("Loaded %d papers from %s", len(all_model_results[model_name]), model_name)

    taxonomy_lookup = _load_taxonomy_lookup(taxonomies_dir) if taxonomies_dir.exists() else {}
    if taxonomy_lookup:
        log.info("Loaded taxonomy lookup for %d taxonomies", len(taxonomy_lookup))

    consensus = aggregate_votes(all_model_results, taxonomy_lookup)

    for paper_filename, paper_data in consensus.items():
        stem = Path(paper_filename).stem
        out_path = consensus_dir / f"{stem}.json"
        out_path.write_text(json.dumps(paper_data, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_aggregate_summary(consensus, all_model_results)
    print(f"Consensus written to: {consensus_dir}")


def _clear_for_reclassify(base_output_dir: Path) -> None:
    if not base_output_dir.exists():
        return
    import shutil
    for child in base_output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            log.info("Removed %s", child)
    print(f"Cleared classification output in {base_output_dir}")

def _prompt_topic_selection(papers_dir: Path, source: str) -> Path:
    default_topic_dir = papers_dir.parent
    papers_base = default_topic_dir.parent
    available = sorted(
        d for d in papers_base.iterdir()
        if d.is_dir() and any((d / source).glob("*.md") or (d / source).glob("*.txt"))
    )
    if not available:
        return default_topic_dir
    default_idx = next((i for i, d in enumerate(available) if d == default_topic_dir), None)
    print("Available topics:")
    for i, d in enumerate(available):
        marker = " (default)" if i == default_idx else ""
        print(f"  [{i + 1}] {d.name}{marker}")
    raw = input(f"Select topic [1-{len(available)}] (Enter = default): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(available):
        return available[int(raw) - 1]
    return default_topic_dir


async def run_classification(cfg: DictConfig) -> None:
    papers_dir = Path(to_absolute_path(cfg.taxonomy.papers_dir))
    source = cfg.taxonomy.source

    topic_dir = _prompt_topic_selection(papers_dir, source)
    papers_dir = topic_dir / source
    taxonomies_dir = topic_dir / "taxonomies"
    base_output_dir = topic_dir / "classification"
    query_name = topic_dir.name.replace("-", " ").replace("_", " ")

    if cfg.taxonomy.get("reclassify", False):
        _clear_for_reclassify(base_output_dir)

    taxonomies = [
        {"name": f.stem, "description": "", "items": json.loads(f.read_text(encoding="utf-8"))}
        for f in sorted(taxonomies_dir.glob("*.json"))
    ]

    taxonomy_lookup = _load_taxonomy_lookup(taxonomies_dir) if taxonomies_dir.exists() else {}

    paper_paths = sorted(list(papers_dir.glob("*.md")) + list(papers_dir.glob("*.txt")))

    log.info("Found %d papers, %d taxonomies", len(paper_paths), len(taxonomies))
    
    # Init NotebookLM Client
    print("\nInitializing NotebookLM Client (reading from ~/.notebooklm/storage_state.json)...")
    try:
        client = await NotebookLMClient.from_storage()
    except Exception as e:
        log.error(f"Failed to auth NotebookLM. Did you run 'notebooklm login'? Error: {e}")
        return
        
    async with client:
        notebook_id = await get_or_create_notebook(client, query_name)
        existing_sources = await sync_sources(client, notebook_id, papers_dir)
        
        output_dir = base_output_dir / "NotebookLM"
        output_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(cfg.taxonomy.get("max_concurrent", 3))

        print(f"\n[NotebookLM] Starting classification of {len(paper_paths)} papers")
        
        tasks = []
        for p in paper_paths:
            source_id = existing_sources.get(p.stem)
            if not source_id:
                log.warning(f"Could not find source id for paper {p.stem}, skipping.")
                continue
            tasks.append(
                 classify_paper(
                     client=client, notebook_id=notebook_id, source_id=source_id,
                     paper_path=p, taxonomies=taxonomies, top_k=cfg.taxonomy.get("top_k", 3),
                     semaphore=semaphore, output_dir=output_dir, skip_existing=cfg.taxonomy.skip_existing,
                     taxonomy_lookup=taxonomy_lookup
                 )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        succeeded = sum(1 for r in results if isinstance(r, dict))
        skipped = sum(1 for r in results if r is None)
        failed = sum(1 for r in results if isinstance(r, Exception))

        print(f"\n[NotebookLM] {succeeded} classified, {skipped} skipped, {failed} failed")
        if failed:
             for r in results:
                 if isinstance(r, Exception):
                     log.error("[NotebookLM] Paper error: %s", r)

    # Aggregate — pass overridden directories directly
    _run_aggregation_from_dirs(base_output_dir, taxonomies_dir)


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(run_classification(cfg))


if __name__ == "__main__":
    main()
