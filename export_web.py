"""Export finished topics as demo projects for the SurveyMaster web app on jahody.github.io.

Each papers/{topic}/ folder with abstracts, taxonomies and classification results becomes one
JSON file in the web app's Project format (see jahody.github.io/src/surveymaster/types.ts), plus
an index.json listing them. Only the standard library is needed:

    python export_web.py                      # writes ../jahody.github.io/public/surveymaster/demo/
    python export_web.py --out some/dir --topics latent-reasoning
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT.parent / "jahody.github.io" / "public" / "surveymaster" / "demo"
SOURCE = "abstracts"

# Mirrors DEFAULT_SETTINGS in the web app; the models are replaced by the ones each topic was run with.
SETTINGS = {
    "refiner": "deepseek/deepseek-v3.2",
    "topK": 3,
    "includeHardNegative": True,
    "concurrency": 3,
    "classifyTemperature": 0.0,
    "classifyMaxTokens": 4096,
    "seed": 42,
    "generationTemperature": 0.5,
    "generationMaxTokens": 16384,
    "refinerTemperature": 0.0,
    "evolutionTemperature": 0.3,
    "evolutionMaxTokens": 8192,
    "chunkSize": 20,
    "chunkOverlap": 5,
    "contentChars": 1200,
}


def find_model_dirs(output_dir: Path) -> list[Path]:
    """Model result folders under classification/ (same rule as 3_classify._find_model_dirs)."""
    dirs = {p.parent for p in output_dir.rglob("*.json") if "_consensus" not in p.parent.parts}
    return sorted(dirs)


def load_paper(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8").strip()
    title = text.split("\n", 1)[0].lstrip("#").strip() or md_path.stem
    published = re.search(r"\*\*Published:\*\*\s*(\d{4})-(\d{2})", text)
    authors = re.search(r"\*\*Authors:\*\*\s*(.+)", text)
    paper = {"id": md_path.stem, "title": title, "content": text}
    if re.match(r"^\d{4}\.\d{4,5}", md_path.stem):
        paper["url"] = f"https://arxiv.org/abs/{md_path.stem}"
    if published:
        paper["year"], paper["month"] = int(published.group(1)), int(published.group(2))
    if authors:
        paper["authors"] = authors.group(1).strip()
    return paper


def keep_result(result: dict) -> dict:
    """Only the fields the web app reads from one model × taxonomy result."""
    return {k: result[k] for k in ("chapters", "chapter", "confidence", "error") if k in result}


def export_topic(topic_dir: Path) -> dict | None:
    papers_dir = topic_dir / SOURCE
    tax_dir = topic_dir / "taxonomies"
    cls_dir = topic_dir / "classification"
    consensus_dir = cls_dir / "_consensus"
    if not (papers_dir.is_dir() and tax_dir.is_dir() and consensus_dir.is_dir()):
        return None

    papers = [load_paper(p) for p in sorted(papers_dir.glob("*.md"))]
    taxonomies = [
        {"name": f.stem, "description": "", "items": json.loads(f.read_text(encoding="utf-8"))}
        for f in sorted(tax_dir.glob("*.json"))
    ]

    results: dict[str, dict] = {}
    for model_dir in find_model_dirs(cls_dir):
        model = model_dir.relative_to(cls_dir).as_posix()
        per_paper = {}
        for f in sorted(model_dir.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            paper_id = data["filename"].removesuffix(".md")
            per_paper[paper_id] = {t: keep_result(r) for t, r in data.get("classifications", {}).items()}
        results[model] = per_paper

    consensus = {}
    for f in sorted(consensus_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        paper_id = data["filename"].removesuffix(".md")
        consensus[paper_id] = {
            t: {"chapter": r["chapter"], "score": r["score"], "votes": [[c, s] for c, s in r["votes"].items()], "reason": r.get("reason", "")}
            for t, r in data.get("classifications", {}).items()
        }

    models = sorted(results)
    return {
        "version": 1,
        "id": f"demo-{topic_dir.name}",
        "topic": topic_dir.name.replace("-", " ").replace("_", " "),
        "source": "demo",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "papers": papers,
        "settings": {"models": models, **SETTINGS, "refiner": models[0] if models else SETTINGS["refiner"]},
        "proposals": [],
        "axes": [],
        "taxonomies": taxonomies,
        "results": results,
        "consensus": consensus,
        "usage": {"prompt": 0, "completion": 0, "cost": 0, "calls": 0},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--topics", nargs="*", help="topic folder names under papers/ (default: all exportable)")
    args = ap.parse_args()

    topic_dirs = [ROOT / "papers" / t for t in args.topics] if args.topics else sorted((ROOT / "papers").iterdir())
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for topic_dir in topic_dirs:
        project = export_topic(topic_dir) if topic_dir.is_dir() else None
        if not project:
            print(f"skip {topic_dir.name} (missing {SOURCE}, taxonomies or consensus)")
            continue
        out = args.out / f"{topic_dir.name}.json"
        out.write_text(json.dumps(project, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        index.append({
            "file": out.name,
            "topic": project["topic"],
            "papers": len(project["papers"]),
            "taxonomies": len(project["taxonomies"]),
            "models": project["settings"]["models"],
        })
        print(f"{topic_dir.name}: {len(project['papers'])} papers, {len(project['results'])} models -> {out} ({out.stat().st_size // 1024} KB)")
    (args.out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
