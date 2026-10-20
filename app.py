from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET

app = Flask(__name__)
TAXONOMY_DIR = Path(__file__).parent / "taxonomy"
GT_DIR = Path(__file__).parent / "outputs" / "taxonomy_classifications" / "GT"
CLASSIFICATIONS_DIR = Path(__file__).parent / "outputs" / "taxonomy_classifications"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
BASE_DIR = Path(__file__).parent
PAPERS_DIR = Path(__file__).parent / "papers"
ALLOWED_DATASETS = {"taxonomy_classifications", "taxonomy_classifications2"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/editor")
def editor():
    return render_template("2_editor.html")


@app.route("/classification")
def classification():
    return render_template("3_classification.html")



@app.route("/evaluation")
def evaluation():
    return render_template("4_evaluation.html")


@app.route("/evolution-tree")
def evolution_tree():
    return render_template("5_tree.html")


@app.route("/papers")
def papers():
    return render_template("1_papers.html")


@app.route("/api/taxonomies")
def list_taxonomies():
    files = sorted(p.name for p in TAXONOMY_DIR.glob("*.json"))
    return jsonify(files)


@app.route("/api/taxonomy/<filename>", methods=["GET"])
def get_taxonomy(filename):
    path = TAXONOMY_DIR / filename
    if not path.exists() or not path.suffix == ".json":
        return jsonify({"error": "File not found"}), 404
    return app.response_class(path.read_text(), mimetype="application/json")


@app.route("/api/taxonomy/<filename>", methods=["POST"])
def save_taxonomy(filename):
    path = TAXONOMY_DIR / filename
    if not path.suffix == ".json":
        return jsonify({"error": "Invalid filename"}), 400
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    # Backup before overwriting
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


# --- Papers folder taxonomies ---


@app.route("/api/paper-folders")
def list_paper_folders():
    if not PAPERS_DIR.is_dir():
        return jsonify([])
    folders = sorted(
        p.name for p in PAPERS_DIR.iterdir()
        if p.is_dir() and (
            (p / "taxonomies").is_dir()
            or (p / "search_results.json").exists()
            or (p / "papers.json").exists()
        )
    )
    return jsonify(folders)


@app.route("/api/paper-folder/<folder>/taxonomies")
def list_paper_folder_taxonomies(folder):
    tax_dir = PAPERS_DIR / folder / "taxonomies"
    if not tax_dir.is_dir() or not tax_dir.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Not found"}), 404
    files = sorted(p.name for p in tax_dir.glob("*.json"))
    return jsonify(files)


@app.route("/api/paper-folder/<folder>/taxonomy/<filename>", methods=["GET", "POST"])
def paper_folder_taxonomy(folder, filename):
    path = PAPERS_DIR / folder / "taxonomies" / filename
    if path.suffix != ".json":
        return jsonify({"error": "Invalid filename"}), 400
    if not path.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    if request.method == "GET":
        if not path.exists():
            return jsonify({"error": "File not found"}), 404
        return app.response_class(path.read_text(), mimetype="application/json")
    # POST - save
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


# --- Evolution trees ---


@app.route("/api/evolution-folders")
def list_evolution_folders():
    if not PAPERS_DIR.is_dir():
        return jsonify([])
    folders = sorted(
        p.name for p in PAPERS_DIR.iterdir()
        if p.is_dir() and (p / "evolution").is_dir()
    )
    return jsonify(folders)


@app.route("/api/paper-folder/<folder>/evolution-trees")
def list_evolution_trees(folder):
    evo_dir = PAPERS_DIR / folder / "evolution"
    if not evo_dir.is_dir() or not evo_dir.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify([])
    files = sorted(p.name for p in evo_dir.glob("*.json"))
    return jsonify(files)


@app.route("/api/paper-folder/<folder>/evolution-tree/<filename>")
def get_evolution_tree(folder, filename):
    path = PAPERS_DIR / folder / "evolution" / filename
    if path.suffix != ".json":
        return jsonify({"error": "Invalid filename"}), 400
    if not path.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return app.response_class(path.read_text(), mimetype="application/json")


# --- Classifications file server ---


@app.route("/api/classifications/<dataset>/<path:filepath>")
def serve_classification_file(dataset, filepath):
    if dataset not in ALLOWED_DATASETS:
        return jsonify({"error": "Invalid dataset"}), 400
    base = OUTPUTS_DIR / dataset
    full_path = (base / filepath).resolve()
    if not full_path.is_relative_to(base.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(base, filepath)


# --- Classification API ---


@app.route("/api/paper-folder/<folder>/taxonomies-with-gt")
def list_taxonomies_with_gt(folder):
    """Return list of taxonomy stems that have a matching *-GT.json in taxonomies_GT/."""
    gt_dir = (PAPERS_DIR / folder / "taxonomies_GT").resolve()
    if not gt_dir.is_relative_to(PAPERS_DIR.resolve()) or not gt_dir.is_dir():
        return jsonify([])
    stems = [jf.stem[:-3] for jf in gt_dir.glob("*-GT.json")]  # strip "-GT"
    return jsonify(sorted(stems))


@app.route("/api/paper-folder/<folder>/taxonomy-gt/<stem>")
def get_taxonomy_gt(folder, stem):
    """Return arxiv→chapter map for a specific taxonomy stem's GT file."""
    gt_dir = (PAPERS_DIR / folder / "taxonomies_GT").resolve()
    if not gt_dir.is_relative_to(PAPERS_DIR.resolve()) or not gt_dir.is_dir():
        return jsonify({}), 404

    gt_file = gt_dir / f"{stem}-GT.json"
    if not gt_file.exists():
        return jsonify({}), 404

    def collect(node, chapter_map):
        for paper in node.get("papers", []):
            arxiv = paper.get("arxiv")
            if arxiv:
                chapter_map[arxiv] = node.get("chapter") or node.get("name", "")
        for child in node.get("items", []):
            collect(child, chapter_map)

    data = json.loads(gt_file.read_text(encoding="utf-8"))
    chapter_map = {}
    if isinstance(data, list):
        for node in data:
            collect(node, chapter_map)
    else:
        collect(data, chapter_map)

    return jsonify(chapter_map)


@app.route("/api/paper-folder/<folder>/manual-gt/<stem>", methods=["GET"])
def get_manual_gt(folder, stem):
    """Return saved manual arxiv→chapter assignments for a taxonomy axis."""
    gt_dir = (PAPERS_DIR / folder / "taxonomies_GT").resolve()
    if not gt_dir.is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({}), 404
    manual_file = gt_dir / f"{stem}-manual.json"
    if not manual_file.exists():
        return jsonify({})
    return jsonify(json.loads(manual_file.read_text(encoding="utf-8")))


@app.route("/api/paper-folder/<folder>/manual-gt/<stem>", methods=["POST"])
def save_manual_gt(folder, stem):
    """Save manual arxiv→chapter assignments for a taxonomy axis."""
    gt_dir = (PAPERS_DIR / folder / "taxonomies_GT").resolve()
    if not gt_dir.is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "invalid folder"}), 400
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "expected object"}), 400
    gt_dir.mkdir(parents=True, exist_ok=True)
    manual_file = gt_dir / f"{stem}-manual.json"
    manual_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/api/gt/files")
def list_gt_files():
    files = sorted(p.name for p in GT_DIR.glob("*.json"))
    return jsonify(files)


@app.route("/api/gt/<filename>", methods=["GET"])
def get_gt_file(filename):
    path = GT_DIR / filename
    if not path.exists() or path.suffix != ".json":
        return jsonify({"error": "File not found"}), 404
    return app.response_class(path.read_text(), mimetype="application/json")


GT_PATH = TAXONOMY_DIR / "taxonomy-GT.json"


def _build_gt_map():
    """Walk taxonomy-GT.json and return {arxiv_id: {chapter, name}}."""
    taxonomy = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt = {}

    def walk(items):
        for item in items:
            for paper in item.get("papers", []):
                gt[paper["arxiv"]] = {
                    "chapter": item.get("chapter") or item.get("name", ""),
                    "name": item.get("name", ""),
                }
            if item.get("items"):
                walk(item["items"])

    for top in taxonomy:
        walk(top.get("items", []))
    return gt


def _load_predictions(path):
    """Extract predicted chapters from an output JSON file.

    Returns a list of dicts: [{"chapter": str, "confidence": int}, ...].
    For old format (single chapter), confidence is 100.
    For new top-k format, confidence comes from the model output.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    predictions = []
    for tax_result in data.get("classifications", {}).values():
        if not isinstance(tax_result, dict):
            continue
        # New top-k format: "chapters" array
        chapters_list = tax_result.get("chapters", [])
        if chapters_list:
            for entry in chapters_list:
                if isinstance(entry, str):
                    try:
                        entry = json.loads(entry)
                    except (json.JSONDecodeError, TypeError):
                        continue
                predictions.append({
                    "chapter": str(entry["chapter"]),
                    "confidence": entry.get("confidence", 0),
                })
        # Old format: single "chapter" field
        elif "chapter" in tax_result:
            ch = tax_result["chapter"]
            if isinstance(ch, list):
                for c in ch:
                    predictions.append({"chapter": str(c), "confidence": 100})
            else:
                predictions.append({"chapter": str(ch), "confidence": 100})
    return predictions


def _has_arxiv_jsons(d: Path) -> bool:
    """Check if a directory contains arxiv-style JSON files."""
    return any(p.stem.replace(".", "").isdigit() for p in d.glob("*.json"))


def _find_model_dirs(outputs_dir=None):
    """Find provider/model dirs that contain arxiv-id JSONs.

    Also picks up _consensus directories (2 levels: category/_consensus/).
    """
    if outputs_dir is None:
        outputs_dir = OUTPUTS_DIR
    dirs = []
    for category in sorted(outputs_dir.iterdir()):
        if not category.is_dir():
            continue
        for provider in sorted(category.iterdir()):
            if not provider.is_dir():
                continue
            # _consensus sits at provider level with JSONs directly inside
            if provider.name == "_consensus" and _has_arxiv_jsons(provider):
                dirs.append(provider)
                continue
            if _has_arxiv_jsons(provider):
                dirs.append(provider)
            for model in sorted(provider.iterdir()):
                if not model.is_dir():
                    continue
                if _has_arxiv_jsons(model):
                    dirs.append(model)
    return dirs


@app.route("/api/paper-folder/<folder>/output-folders")
def list_topic_output_folders(folder):
    """List classification_* dirs inside papers/<folder>/."""
    topic_dir = (BASE_DIR / "papers" / folder).resolve()
    if not topic_dir.is_relative_to(BASE_DIR.resolve()) or not topic_dir.is_dir():
        return jsonify([])
    return jsonify(sorted(
        d.name for d in topic_dir.iterdir()
        if d.is_dir() and d.name.startswith("classification")
    ))


@app.route("/api/output-folders")
def list_output_folders():
    """List classification_* dirs found inside any papers/<topic>/ directory."""
    folders = set()
    papers_dir = BASE_DIR / "papers"
    if papers_dir.is_dir():
        for topic_dir in papers_dir.iterdir():
            if topic_dir.is_dir():
                for d in topic_dir.iterdir():
                    if d.is_dir() and d.name.startswith("classification"):
                        folders.add(d.name)
    return jsonify(sorted(folders))


@app.route("/api/output-folder/<folder>/topics")
def list_output_folder_topics(folder):
    """List topic dirs under papers/ that contain the given classification folder."""
    papers_dir = BASE_DIR / "papers"
    if not papers_dir.is_dir():
        return jsonify([])
    topics = sorted(
        p.name for p in papers_dir.iterdir()
        if p.is_dir() and (p / folder).is_dir()
    )
    return jsonify(topics)


@app.route("/api/output-folder/<folder>/topic/<topic>/classifications")
def get_topic_classifications(folder, topic):
    base = (BASE_DIR / "papers" / topic / folder).resolve()
    if not base.is_relative_to(BASE_DIR.resolve()) or not base.is_dir():
        return jsonify({"error": "not found"}), 404

    papers = {}
    axes_set = set()
    models = []

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "_consensus":
            for jf in sorted(entry.glob("*.json")):
                arxiv = jf.stem
                data = json.loads(jf.read_text(encoding="utf-8"))
                p = papers.setdefault(arxiv, {"models": {}, "consensus": {}})
                for ax, info in data.get("classifications", {}).items():
                    axes_set.add(ax)
                    p["consensus"][ax] = {
                        "chapter": info.get("chapter"),
                        "score": info.get("score"),
                    }
        else:
            jsons = list(entry.glob("*.json"))
            if jsons:
                label = entry.name
                if label not in models:
                    models.append(label)
                for jf in sorted(jsons):
                    arxiv = jf.stem
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    p = papers.setdefault(arxiv, {"models": {}, "consensus": {}})
                    p["models"][label] = {}
                    for ax, info in data.get("classifications", {}).items():
                        axes_set.add(ax)
                        p["models"][label][ax] = [
                            {"chapter": c.get("chapter"), "confidence": c.get("confidence")}
                            for c in info.get("chapters", [])
                        ]
            for model_dir in sorted(entry.iterdir()):
                if not model_dir.is_dir():
                    continue
                label = model_dir.name
                if label not in models:
                    models.append(label)
                for jf in sorted(model_dir.glob("*.json")):
                    arxiv = jf.stem
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    p = papers.setdefault(arxiv, {"models": {}, "consensus": {}})
                    p["models"][label] = {}
                    for ax, info in data.get("classifications", {}).items():
                        axes_set.add(ax)
                        p["models"][label][ax] = [
                            {"chapter": c.get("chapter"), "confidence": c.get("confidence")}
                            for c in info.get("chapters", [])
                        ]

    return jsonify({"papers": papers, "models": models, "axes": sorted(axes_set)})


@app.route("/api/compare-gt-results")
def compare_gt_results():
    """Compare per-paper JSONs in each model dir against taxonomy-GT.json."""
    if not GT_PATH.exists():
        return jsonify([])

    folder = request.args.get("folder", "outputs")
    outputs_dir = (BASE_DIR / folder).resolve()
    if not outputs_dir.is_relative_to(BASE_DIR.resolve()) or not outputs_dir.is_dir():
        return jsonify({"error": "Invalid folder"}), 400

    gt_map = _build_gt_map()
    gt_arxivs = set(gt_map.keys())
    results = []

    for model_dir in _find_model_dirs(outputs_dir):
        label = model_dir.relative_to(outputs_dir).as_posix()
        papers = []
        correct_top1 = 0
        correct_topk = 0
        total = 0

        for path in sorted(model_dir.glob("*.json")):
            arxiv_id = path.stem
            if arxiv_id not in gt_arxivs:
                continue
            predictions = _load_predictions(path)
            gt = gt_map[arxiv_id]
            expected = {gt["chapter"], gt["name"]}
            total += 1

            # Top-1: only the highest-confidence prediction
            top1_match = bool(predictions and predictions[0]["chapter"] in expected)
            # Top-k: any prediction matches
            topk_match = any(p["chapter"] in expected for p in predictions)

            if top1_match:
                correct_top1 += 1
            if topk_match:
                correct_topk += 1

            papers.append({
                "arxiv": arxiv_id,
                "predictions": [
                    {"chapter": p["chapter"], "confidence": p["confidence"]}
                    for p in predictions
                ],
                "expected": gt["chapter"],
                "expected_name": gt["name"],
                "match_top1": top1_match,
                "match_topk": topk_match,
            })

        if total > 0:
            results.append({
                "label": label,
                "accuracy": correct_top1 / total,
                "accuracy_topk": correct_topk / total,
                "correct": correct_top1,
                "correct_topk": correct_topk,
                "total": total,
                "papers": papers,
            })

    return jsonify(results)


@app.route("/api/gt/<filename>", methods=["POST"])
def save_gt_file(filename):
    path = GT_DIR / filename
    if path.suffix != ".json":
        return jsonify({"error": "Invalid filename"}), 400
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})


@app.route("/api/evolution-tree")
def evolution_tree_data():
    folder = request.args.get("folder", "outputs_07")
    outputs_dir = (BASE_DIR / folder).resolve()
    if not outputs_dir.is_relative_to(BASE_DIR.resolve()) or not outputs_dir.is_dir():
        return jsonify({"error": "Invalid folder"}), 400

    # Walk taxonomy-GT to build lanes and paper name map
    taxonomy_gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    lanes = []
    paper_names = {}

    def walk_taxonomy(items, parent_chapter=None):
        for item in items:
            ch = item.get("chapter") or item.get("name", "")
            has_children = bool(item.get("items"))
            for paper in item.get("papers", []):
                paper_names[paper["arxiv"]] = paper.get("name", "")
            if not has_children:
                lanes.append({
                    "chapter": ch,
                    "name": item["name"],
                    "parent": parent_chapter,
                })
            if item.get("items"):
                walk_taxonomy(item["items"], ch)

    for top in taxonomy_gt:
        walk_taxonomy(top.get("items", []))

    # Find consensus dirs and load classifications
    papers = []
    for category in sorted(outputs_dir.iterdir()):
        if not category.is_dir():
            continue
        consensus_path = category / "_consensus"
        if not consensus_path.is_dir():
            continue
        for path in sorted(consensus_path.glob("*.json")):
            arxiv_id = path.stem
            data = json.loads(path.read_text(encoding="utf-8"))
            cls = data.get("classifications", {})
            for tax_data in cls.values():
                if not isinstance(tax_data, dict):
                    continue
                chapter = tax_data.get("chapter", "")
                score = tax_data.get("score", 0)
                votes = tax_data.get("votes", {})
                reason = tax_data.get("reason", "")
                numeric_votes = {
                    k: v for k, v in votes.items() if k[:1].isdigit()
                }
                total = sum(numeric_votes.values())
                top_vote = max(numeric_votes.values(), default=0)
                papers.append({
                    "arxiv": arxiv_id,
                    "name": paper_names.get(arxiv_id, arxiv_id),
                    "chapter": chapter,
                    "score": score,
                    "unanimity": round(top_vote / total, 3) if total else 0,
                    "reason": reason,
                    "votes": numeric_votes,
                })

    return jsonify({"lanes": lanes, "papers": papers})


# --- Papers management API ---


@app.route("/api/paper-folder/<folder>/papers", methods=["GET"])
def list_papers(folder):
    path = PAPERS_DIR / folder / "search_results.json"
    if not path.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    if path.exists():
        return app.response_class(path.read_text(encoding="utf-8"), mimetype="application/json")
    fallback = PAPERS_DIR / folder / "papers.json"
    if not fallback.exists():
        return jsonify([])
    items = json.loads(fallback.read_text(encoding="utf-8"))
    converted = [
        {"externalIds": {"ArXiv": p["arxiv"]}, "title": p["name"], "year": "", "authors": []}
        for p in items if "arxiv" in p
    ]
    return jsonify(converted)


@app.route("/api/paper-folder/<folder>/papers", methods=["POST"])
def add_paper(folder):
    path = PAPERS_DIR / folder / "search_results.json"
    if not path.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json()
    if not data or "arxiv_id" not in data:
        return jsonify({"error": "arxiv_id is required"}), 400

    arxiv_id = data["arxiv_id"].strip()
    papers = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    # Check for duplicates
    for p in papers:
        ext = p.get("externalIds", {})
        if ext.get("ArXiv") == arxiv_id:
            return jsonify({"error": f"Paper {arxiv_id} already exists"}), 409

    # Fetch metadata from arXiv API
    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "SurveyMaster/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch from arXiv: {e}"}), 502

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    entry = root.find("atom:entry", ns)
    if entry is None:
        return jsonify({"error": f"No paper found for arxiv ID {arxiv_id}"}), 404

    title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
    if not title or title.startswith("Error"):
        return jsonify({"error": f"No paper found for arxiv ID {arxiv_id}"}), 404

    authors = [
        {"name": a.findtext("atom:name", "", ns)}
        for a in entry.findall("atom:author", ns)
    ]
    published = entry.findtext("atom:published", "", ns)
    year = int(published[:4]) if len(published) >= 4 else None

    new_paper = {
        "externalIds": {"ArXiv": arxiv_id},
        "title": title,
        "year": year,
        "authors": authors,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }

    # Backup and save
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    papers.append(new_paper)
    path.write_text(json.dumps(papers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return jsonify(new_paper), 201


@app.route("/api/paper-folder/<folder>/papers/<arxiv_id>", methods=["DELETE"])
def remove_paper(folder, arxiv_id):
    path = PAPERS_DIR / folder / "search_results.json"
    if not path.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    if not path.exists():
        return jsonify({"error": "File not found"}), 404

    papers = json.loads(path.read_text(encoding="utf-8"))
    original_len = len(papers)
    papers = [
        p for p in papers
        if p.get("externalIds", {}).get("ArXiv") != arxiv_id
    ]
    if len(papers) == original_len:
        return jsonify({"error": f"Paper {arxiv_id} not found"}), 404

    shutil.copy2(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(papers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/api/paper-folder/<folder>/evaluation")
def get_evaluation_metrics(folder):
    path = PAPERS_DIR / folder / "evaluation" / "consistency_metrics.json"
    if not path.resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    if not path.exists():
        return jsonify({"error": "No evaluation data. Run: python 3d_evaluate.py"}), 404
    return app.response_class(path.read_text(encoding="utf-8"), mimetype="application/json")


@app.route("/api/paper-folder/<folder>/evaluation/plots/<filename>")
def get_evaluation_plot(folder, filename):
    plot_dir = PAPERS_DIR / folder / "evaluation" / "plots"
    if not (plot_dir / filename).resolve().is_relative_to(PAPERS_DIR.resolve()):
        return jsonify({"error": "Forbidden"}), 403
    return send_from_directory(plot_dir, filename)


if __name__ == "__main__":
    app.run(port=8765, debug=True)
