"""Step 3: Classify papers against taxonomies, then aggregate via confidence voting."""

import asyncio
import json
import logging
import re
from collections import defaultdict
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

def _load_prompts(filepath: Path) -> tuple[str, str]:
    if not filepath.exists():
        return "", ""
    text = filepath.read_text(encoding="utf-8")
    sys_prompt, usr_prompt = "", ""
    # Split only on lines that are exactly "# Title" (top-level headers at line start)
    import re
    sections = re.split(r'^# ', text, flags=re.MULTILINE)
    for part in sections:
        if part.startswith("System Prompt"):
            sys_prompt = part.split("\n", 1)[1].strip()
        elif part.startswith("User Prompt"):
            usr_prompt = part.split("\n", 1)[1].strip()
    return sys_prompt, usr_prompt

DEFAULT_PROMPTS_FILE = Path(__file__).parent / "prompts" / "3_prompt.md"
SYSTEM_MESSAGE, USER_PROMPT_TEMPLATE = _load_prompts(DEFAULT_PROMPTS_FILE)


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


def build_prompt(paper_content: str, taxonomy: dict, filename: str, top_k: int = 3, include_hard_negative: bool = True) -> str:
    """Build the user message for classifying one paper against one taxonomy."""
    leaves = collect_leaf_items(taxonomy["items"])
    items_text = ""
    for item in leaves:
        chapter = item.get("chapter", "unknown")
        items_text += f'\n### {item.get("name", "Unknown")} (chapter {chapter})\n'
        items_text += f'**Description:** {item.get("description", "")}\n'
        items_text += f'**Key phrases:** {", ".join(item.get("phrases", []))}\n'
        if include_hard_negative:
            items_text += f'**Hard negative (DO NOT assign if this describes the paper\'s approach):** {item.get("hard_negative", "")}\n'

    return USER_PROMPT_TEMPLATE.format(
        paper_content=paper_content,
        top_k=top_k,
        taxonomy_name=taxonomy.get("name", ""),
        taxonomy_description=taxonomy.get("description", ""),
        items_text=items_text,
        filename=filename,
    )


def parse_json_response(text: str | None) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    if not text:
        raise ValueError("Empty or null response from model")
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


async def classify_one(
    client: AsyncOpenAI,
    paper_content: str,
    taxonomy: dict,
    filename: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    top_k: int,
    semaphore: asyncio.Semaphore,
    token_usage: dict,
    include_hard_negative: bool = True,
    system_message: str | None = None,
    user_prompt_template: str | None = None,
) -> dict:
    """Classify a single paper against a single taxonomy (one LLM call)."""
    prompt = build_prompt(paper_content, taxonomy, filename, top_k, include_hard_negative)
    sys_msg = system_message or SYSTEM_MESSAGE
    usr_tpl = user_prompt_template

    for attempt in range(3):
        async with semaphore:
            try:
                kwargs = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                if seed is not None:
                    kwargs["seed"] = seed
                response = await client.chat.completions.create(**kwargs)
                if response.usage:
                    token_usage["prompt"] += response.usage.prompt_tokens
                    token_usage["completion"] += response.usage.completion_tokens
                content = response.choices[0].message.content
                return parse_json_response(content)
            except (RateLimitError, APITimeoutError) as e:
                delay = 2 ** (attempt + 1)
                log.warning(
                    "Retry %d/3 for taxonomy '%s': %s (waiting %ds)",
                    attempt + 1, taxonomy["name"], e, delay,
                )
                await asyncio.sleep(delay)
            except json.JSONDecodeError as e:
                if attempt < 2:
                    log.warning(
                        "JSON parse error for taxonomy '%s', retrying: %s",
                        taxonomy["name"], e,
                    )
                    await asyncio.sleep(1)
                else:
                    log.error("Failed to parse JSON for taxonomy '%s'", taxonomy["name"])
                    return {"error": f"JSON parse error: {e}"}
            except Exception as e:
                log.error("Error for taxonomy '%s': %s", taxonomy["name"], e)
                return {"error": str(e)}

    return {"error": "Max retries exceeded"}


async def classify_paper(
    client: AsyncOpenAI,
    paper_path: Path,
    taxonomies: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    top_k: int,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    skip_existing: bool,
    token_usage: dict,
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
    include_hard_negative: bool = True,
    system_message: str | None = None,
    user_prompt_template: str | None = None,
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

    paper_content = paper_path.read_text(encoding="utf-8")
    log.info("Classifying %s (running %d/%d taxonomies) ...", paper_path.name, len(taxonomies_to_run), len(taxonomies))

    tasks = [
        classify_one(
            client, paper_content, tax, paper_path.name,
            model, temperature, max_tokens, seed, top_k, semaphore, token_usage,
            include_hard_negative, system_message, user_prompt_template,
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


async def run_single_model(
    model: str,
    paper_paths: list[Path],
    taxonomies: list[dict],
    cfg: DictConfig,
    base_output_dir: Path,
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
    pricing_map: dict[str, tuple[float, float]] | None = None,
    include_hard_negative: bool = True,
    system_message: str | None = None,
    user_prompt_template: str | None = None,
) -> dict:
    """Run classification for all papers with a single model. Returns cost info."""
    output_dir = base_output_dir / model
    output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(
        api_key=cfg.taxonomy.api_key,
        base_url=cfg.taxonomy.base_url,
    )
    semaphore = asyncio.Semaphore(cfg.taxonomy.max_concurrent)
    token_usage = {"prompt": 0, "completion": 0}
    pricing = (pricing_map or {}).get(model)

    log.info("[%s] Starting classification of %d papers", model, len(paper_paths))

    tasks = [
        classify_paper(
            client, p, taxonomies,
            model, cfg.taxonomy.temperature,
            cfg.taxonomy.max_tokens, cfg.taxonomy.get("seed"),
            cfg.taxonomy.get("top_k", 3), semaphore, output_dir,
            cfg.taxonomy.skip_existing, token_usage, taxonomy_lookup,
            include_hard_negative, system_message, user_prompt_template,
        )
        for p in paper_paths
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    succeeded = sum(1 for r in results if isinstance(r, dict))
    skipped = sum(1 for r in results if r is None)
    failed = sum(1 for r in results if isinstance(r, Exception))

    slug = model.split("/")[-1]
    print(f"\n[{slug}] {succeeded} classified, {skipped} skipped, {failed} failed")
    cost = print_cost(slug, token_usage, pricing)
    if failed:
        for r in results:
            if isinstance(r, Exception):
                log.error("[%s] Paper error: %s", model, r)
    return {"cost": cost, "prompt": token_usage["prompt"], "completion": token_usage["completion"], "has_pricing": pricing is not None}


# ---------------------------------------------------------------------------
# Aggregation (confidence-weighted majority voting)
# ---------------------------------------------------------------------------

def _build_name_to_chapter(items: list[dict], mapping: dict) -> None:
    """Recursively populate mapping from item name -> chapter number."""
    for item in items:
        if "name" in item and "chapter" in item:
            mapping[item["name"]] = str(item["chapter"])
        if "items" in item:
            _build_name_to_chapter(item["items"], mapping)


def _load_taxonomy_lookup(taxonomies_dir: Path) -> dict[str, dict[str, str]]:
    """Return {taxonomy_stem: {item_name: chapter_number}} for all taxonomy files."""
    lookup: dict[str, dict[str, str]] = {}
    for json_path in sorted(taxonomies_dir.glob("*.json")):
        mapping: dict[str, str] = {}
        items = json.loads(json_path.read_text(encoding="utf-8"))
        _build_name_to_chapter(items, mapping)
        lookup[json_path.stem] = mapping
    return lookup


def _find_model_dirs(output_dir: Path) -> list[Path]:
    """Find all model result directories (contain .json files, skip _consensus)."""
    model_dirs = []
    for json_file in output_dir.rglob("*.json"):
        model_dir = json_file.parent
        if "_consensus" not in model_dir.parts and model_dir not in model_dirs:
            model_dirs.append(model_dir)
    return sorted(model_dirs)


def _load_classifications(model_dir: Path) -> dict[str, dict]:
    """Load all classification JSONs from a model directory, keyed by filename."""
    results = {}
    for json_path in sorted(model_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        results[data["filename"]] = data
    return results


def _normalize_chapter(chapter: str, tax_name: str, taxonomy_lookup: dict[str, dict[str, str]]) -> str:
    """If chapter is an item name, return its chapter number; otherwise return as-is."""
    name_map = taxonomy_lookup.get(tax_name, {})
    # Direct name lookup
    if chapter in name_map:
        return name_map[chapter]
    # Extract chapter number from patterns like "Name (chapter 3.3)"
    m = re.search(r'\(chapter\s+([\d.]+(?:\([a-z]\))?)\)', chapter)
    if m:
        return m.group(1)
    return chapter


def _normalize_result(classifications: dict, taxonomy_lookup: dict[str, dict[str, str]]) -> None:
    """Replace chapter names with chapter numbers in-place across all taxonomy results."""
    for tax_name, tax_result in classifications.items():
        if "error" in tax_result:
            continue
        for entry in tax_result.get("chapters", []):
            if isinstance(entry, dict) and "chapter" in entry:
                entry["chapter"] = _normalize_chapter(str(entry["chapter"]), tax_name, taxonomy_lookup)


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
    """Print summary of voting results."""
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
    """Aggregate classifications from all models into consensus."""
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


def run_aggregation(cfg: DictConfig) -> None:
    """Aggregate classifications from all models into consensus."""
    source_dir = Path(to_absolute_path(cfg.taxonomy.aggregate_dir))
    taxonomies_dir = Path(to_absolute_path(cfg.taxonomy.taxonomies_dir))
    _run_aggregation_from_dirs(source_dir, taxonomies_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _clear_for_reclassify(base_output_dir: Path) -> None:
    """Delete all existing classification results (per-model + consensus)."""
    if not base_output_dir.exists():
        return
    import shutil
    for child in base_output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            log.info("Removed %s", child)
    print(f"Cleared classification output in {base_output_dir}")


def _prompt_topic_selection(papers_dir: Path, source: str, auto_topic: bool = False) -> Path:
    """Let user pick a topic dir at runtime. Returns the chosen topic dir."""
    default_topic_dir = papers_dir.parent
    if auto_topic:
        return default_topic_dir
    papers_base = default_topic_dir.parent
    available = sorted(
        d for d in papers_base.iterdir()
        if d.is_dir() and any((d / source).glob("*.md"))
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
    """Load data, classify all papers with all models, then aggregate."""
    papers_dir = Path(to_absolute_path(cfg.taxonomy.papers_dir))
    source = cfg.taxonomy.source

    topic_dir = _prompt_topic_selection(papers_dir, source, cfg.taxonomy.get("auto_topic", False))
    papers_dir = topic_dir / source
    taxonomies_dir = topic_dir / "taxonomies"
    
    if cfg.taxonomy.get("output_dir_override"):
        base_output_dir = topic_dir / cfg.taxonomy.output_dir_override
    else:
        base_output_dir = topic_dir / "classification"

    if cfg.taxonomy.get("reclassify", False):
        _clear_for_reclassify(base_output_dir)

    taxonomies = [
        {"name": f.stem, "description": "", "items": json.loads(f.read_text(encoding="utf-8"))}
        for f in sorted(taxonomies_dir.glob("*.json"))
    ]

    taxonomy_lookup: dict[str, dict[str, str]] = {}
    for tax in taxonomies:
        mapping: dict[str, str] = {}
        _build_name_to_chapter(tax["items"], mapping)
        taxonomy_lookup[tax["name"]] = mapping

    paper_paths = sorted(papers_dir.glob("*.md"))

    models = list(cfg.taxonomy.get("models") or [cfg.taxonomy.model])
    log.info("Found %d papers, %d taxonomies, %d models", len(paper_paths), len(taxonomies), len(models))
    log.info("Models: %s", ", ".join(models))

    # Load custom prompts if specified
    prompts_file_str = cfg.taxonomy.get("prompts_file", None)
    custom_sys, custom_usr = None, None
    if prompts_file_str:
        pf = Path(to_absolute_path(prompts_file_str))
        custom_sys, custom_usr = _load_prompts(pf)
        log.info("Using custom prompts from %s", pf)

    include_hard_negative = cfg.taxonomy.get("include_hard_negative", True)

    pricing_map = await fetch_all_pricing(cfg.taxonomy.base_url, cfg.taxonomy.api_key)
    model_results = await asyncio.gather(*[
        run_single_model(model, paper_paths, taxonomies, cfg, base_output_dir, taxonomy_lookup, pricing_map,
                         include_hard_negative, custom_sys, custom_usr)
        for model in models
    ], return_exceptions=True)

    print(f"\nAll {len(models)} model(s) classified.")
    total_cost = sum(r["cost"] for r in model_results if isinstance(r, dict))
    total_prompt = sum(r["prompt"] for r in model_results if isinstance(r, dict))
    total_completion = sum(r["completion"] for r in model_results if isinstance(r, dict))
    any_missing = any(not r["has_pricing"] for r in model_results if isinstance(r, dict))
    if total_prompt + total_completion:
        print(f"Total: {total_prompt:,} in / {total_completion:,} out / {total_prompt + total_completion:,} tokens", end="")
        if total_cost:
            suffix = " (partial)" if any_missing else ""
            print(f"  ${total_cost:.4f}{suffix}")
        elif any_missing:
            print("  (cost unavailable)")
        else:
            print()

    # Aggregate — pass overridden dirs directly
    _run_aggregation_from_dirs(base_output_dir, taxonomies_dir)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    asyncio.run(run_classification(cfg))


if __name__ == "__main__":
    main()
