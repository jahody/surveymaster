"""Step 4: Evaluate classification quality, optionally refine taxonomies.

Runs all consistency/coverage metrics, shows results, then asks whether to
auto-refine imbalanced taxonomies. If refined, user reviews before reclassifying.
"""

import asyncio
import json
import logging
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev

import httpx

import _resolvers  # noqa: F401 – registers OmegaConf resolvers
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from openai import AsyncOpenAI, RateLimitError, APITimeoutError

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def build_name_to_chapter(items: list[dict], mapping: dict) -> None:
    for item in items:
        if "name" in item and "chapter" in item:
            mapping[item["name"]] = str(item["chapter"])
        if "items" in item:
            build_name_to_chapter(item["items"], mapping)


def load_taxonomy_lookup(taxonomies_dir: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for json_path in sorted(taxonomies_dir.glob("*.json")):
        mapping: dict[str, str] = {}
        items = json.loads(json_path.read_text(encoding="utf-8"))
        build_name_to_chapter(items, mapping)
        lookup[json_path.stem] = mapping
    return lookup


def find_model_dirs(output_dir: Path) -> list[Path]:
    model_dirs = []
    for json_file in output_dir.rglob("*.json"):
        model_dir = json_file.parent
        if "_consensus" not in model_dir.parts and model_dir not in model_dirs:
            model_dirs.append(model_dir)
    return sorted(model_dirs)


def load_classifications(model_dir: Path) -> dict[str, dict]:
    results = {}
    for json_path in sorted(model_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        results[data["filename"]] = data
    return results


def normalize_chapter(chapter: str, tax_name: str, taxonomy_lookup: dict) -> str:
    name_map = taxonomy_lookup.get(tax_name, {})
    return name_map.get(chapter, chapter)


# ---------------------------------------------------------------------------
# Data normalization
# ---------------------------------------------------------------------------

def extract_normalized_predictions(
    all_model_results: dict[str, dict[str, dict]],
    taxonomy_lookup: dict[str, dict[str, str]],
) -> dict[str, dict[str, dict[str, list[dict]]]]:
    predictions: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for model_name, model_results in all_model_results.items():
        predictions[model_name] = {}
        for paper_fn, paper_data in model_results.items():
            predictions[model_name][paper_fn] = {}
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
                normalized = []
                for entry in chapters:
                    if isinstance(entry, dict):
                        ch_str = str(entry["chapter"])
                        try:
                            conf = int(entry.get("confidence", 100))
                        except (ValueError, TypeError):
                            conf = 50
                    else:
                        ch_str = str(entry)
                        conf = 100 // max(len(chapters), 1)
                    ch_str = normalize_chapter(ch_str, tax_name, taxonomy_lookup)
                    normalized.append({"chapter": ch_str, "confidence": conf})
                predictions[model_name][paper_fn][tax_name] = normalized
    return predictions


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    all_labels = set(labels_a) | set(labels_b)
    p_e = sum((labels_a.count(l) / n) * (labels_b.count(l) / n) for l in all_labels)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(ratings_matrix: list[list[int]], n_raters: int) -> float:
    n_items = len(ratings_matrix)
    if n_items == 0 or n_raters <= 1:
        return 0.0
    n_cats = len(ratings_matrix[0])
    p_items = []
    for row in ratings_matrix:
        p_i = (sum(n_ij * n_ij for n_ij in row) - n_raters) / (n_raters * (n_raters - 1))
        p_items.append(p_i)
    p_bar = sum(p_items) / n_items
    p_j = []
    for j in range(n_cats):
        total_j = sum(row[j] for row in ratings_matrix)
        p_j.append(total_j / (n_items * n_raters))
    p_e_bar = sum(pj * pj for pj in p_j)
    if p_e_bar >= 1.0:
        return 1.0
    return (p_bar - p_e_bar) / (1.0 - p_e_bar)


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Metrics (same as 3d_evaluate.py)
# ---------------------------------------------------------------------------

def compute_pairwise_agreement(predictions: dict) -> dict:
    models = sorted(predictions.keys())
    all_papers = set()
    for m in models:
        all_papers.update(predictions[m].keys())
    all_taxonomies = set()
    for m in models:
        for p in predictions[m]:
            all_taxonomies.update(predictions[m][p].keys())

    pairs = {}
    for m_a, m_b in combinations(models, 2):
        labels_a, labels_b = [], []
        per_tax = defaultdict(lambda: {"a": [], "b": []})
        for paper in sorted(all_papers):
            for tax in sorted(all_taxonomies):
                chs_a = predictions.get(m_a, {}).get(paper, {}).get(tax)
                chs_b = predictions.get(m_b, {}).get(paper, {}).get(tax)
                if not chs_a or not chs_b:
                    continue
                top_a = chs_a[0]["chapter"]
                top_b = chs_b[0]["chapter"]
                labels_a.append(top_a)
                labels_b.append(top_b)
                per_tax[tax]["a"].append(top_a)
                per_tax[tax]["b"].append(top_b)

        n = len(labels_a)
        agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
        pair_key = f"{m_a} vs {m_b}"
        pairs[pair_key] = {
            "agreement_rate": round(agree / n, 4) if n else 0,
            "cohens_kappa": round(cohens_kappa(labels_a, labels_b), 4),
            "n_compared": n,
            "per_taxonomy": {},
        }
        for tax in sorted(per_tax):
            a_list = per_tax[tax]["a"]
            b_list = per_tax[tax]["b"]
            t_n = len(a_list)
            t_agree = sum(1 for a, b in zip(a_list, b_list) if a == b)
            pairs[pair_key]["per_taxonomy"][tax] = {
                "agreement_rate": round(t_agree / t_n, 4) if t_n else 0,
                "cohens_kappa": round(cohens_kappa(a_list, b_list), 4),
                "n": t_n,
            }

    matrix = [[0.0] * len(models) for _ in models]
    for i, m_i in enumerate(models):
        for j, m_j in enumerate(models):
            if i == j:
                matrix[i][j] = 1.0
            elif i < j:
                key = f"{m_i} vs {m_j}"
                matrix[i][j] = pairs[key]["agreement_rate"]
                matrix[j][i] = pairs[key]["agreement_rate"]

    return {"pairs": pairs, "matrix": matrix, "model_names": models}


def compute_fleiss_kappa(predictions: dict) -> dict:
    models = sorted(predictions.keys())
    n_raters = len(models)
    all_papers = set()
    for m in models:
        all_papers.update(predictions[m].keys())
    all_taxonomies = set()
    for m in models:
        for p in predictions[m]:
            all_taxonomies.update(predictions[m][p].keys())

    per_tax_kappa = {}
    all_ratings = []
    all_categories = set()
    for paper in all_papers:
        for tax in all_taxonomies:
            for m in models:
                chs = predictions.get(m, {}).get(paper, {}).get(tax)
                if chs:
                    all_categories.add(chs[0]["chapter"])

    cat_list = sorted(all_categories)
    cat_idx = {c: i for i, c in enumerate(cat_list)}

    for tax in sorted(all_taxonomies):
        tax_ratings = []
        for paper in sorted(all_papers):
            row = [0] * len(cat_list)
            raters_for_item = 0
            for m in models:
                chs = predictions.get(m, {}).get(paper, {}).get(tax)
                if chs and chs[0]["chapter"] in cat_idx:
                    row[cat_idx[chs[0]["chapter"]]] += 1
                    raters_for_item += 1
            if raters_for_item == n_raters:
                tax_ratings.append(row)
                all_ratings.append(row)
        if tax_ratings:
            per_tax_kappa[tax] = round(fleiss_kappa(tax_ratings, n_raters), 4)

    overall = round(fleiss_kappa(all_ratings, n_raters), 4) if all_ratings else 0.0
    return {"overall": overall, "per_taxonomy": per_tax_kappa, "n_raters": n_raters, "n_items_overall": len(all_ratings)}


def compute_consensus_strength(consensus_dir: Path) -> dict:
    margins = []
    per_tax: dict[str, list[float]] = defaultdict(list)
    per_paper: dict[str, dict] = {}
    for json_path in sorted(consensus_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        paper_fn = data["filename"]
        per_paper[paper_fn] = {}
        for tax_name, tax_result in data.get("classifications", {}).items():
            votes = tax_result.get("votes", {})
            sorted_votes = sorted(votes.values(), reverse=True)
            winner_score = sorted_votes[0] if sorted_votes else 0
            runner_up = sorted_votes[1] if len(sorted_votes) > 1 else 0
            margin = winner_score - runner_up
            margins.append(margin)
            per_tax[tax_name].append(margin)
            per_paper[paper_fn][tax_name] = {"margin": margin, "winner": tax_result.get("chapter", ""), "winner_score": winner_score}

    def margin_stats(vals):
        if not vals:
            return {}
        return {"mean": round(mean(vals), 1), "median": round(median(vals), 1), "std": round(stdev(vals), 1) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals), "weak_count": sum(1 for v in vals if v < 20), "strong_count": sum(1 for v in vals if v >= 20), "n": len(vals)}

    return {"overall": margin_stats(margins), "per_taxonomy": {t: margin_stats(v) for t, v in sorted(per_tax.items())}, "per_paper": per_paper}


def compute_confidence_agreement(predictions: dict) -> dict:
    models = sorted(predictions.keys())
    all_papers = set()
    for m in models:
        all_papers.update(predictions[m].keys())
    all_taxonomies = set()
    for m in models:
        for p in predictions[m]:
            all_taxonomies.update(predictions[m][p].keys())

    points = []
    for paper in sorted(all_papers):
        for tax in sorted(all_taxonomies):
            top1s, confs = [], []
            for m in models:
                chs = predictions.get(m, {}).get(paper, {}).get(tax)
                if chs:
                    top1s.append(chs[0]["chapter"])
                    confs.append(chs[0]["confidence"])
            if len(top1s) < 2:
                continue
            most_common = Counter(top1s).most_common(1)[0]
            agreement = most_common[1] / len(top1s)
            points.append({"paper": paper, "taxonomy": tax, "agreement": round(agreement, 4), "mean_confidence": round(mean(confs), 1)})

    agreements = [p["agreement"] for p in points]
    confidences = [p["mean_confidence"] for p in points]
    r = pearson_r(agreements, confidences) if points else 0.0
    return {"pearson_r": round(r, 4), "n_points": len(points), "points": points}


def compute_model_deviation(predictions: dict, consensus_dir: Path) -> dict:
    consensus_winners: dict[str, dict[str, str]] = {}
    for json_path in sorted(consensus_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        paper_fn = data["filename"]
        consensus_winners[paper_fn] = {}
        for tax_name, tax_result in data.get("classifications", {}).items():
            consensus_winners[paper_fn][tax_name] = tax_result.get("chapter", "")

    models = sorted(predictions.keys())
    model_stats = {}
    for m in models:
        deviations = 0
        total = 0
        per_tax: dict[str, dict] = defaultdict(lambda: {"deviations": 0, "total": 0})
        for paper_fn, taxonomies in predictions[m].items():
            for tax_name, chs in taxonomies.items():
                winner = consensus_winners.get(paper_fn, {}).get(tax_name)
                if winner is None or not chs:
                    continue
                total += 1
                per_tax[tax_name]["total"] += 1
                if chs[0]["chapter"] != winner:
                    deviations += 1
                    per_tax[tax_name]["deviations"] += 1
        model_stats[m] = {
            "deviation_rate": round(deviations / total, 4) if total else 0,
            "deviations": deviations, "total": total,
            "per_taxonomy": {t: {"deviation_rate": round(v["deviations"] / v["total"], 4) if v["total"] else 0, **v} for t, v in sorted(per_tax.items())},
        }
    outlier = max(model_stats, key=lambda m: model_stats[m]["deviation_rate"]) if model_stats else None
    return {"models": model_stats, "outlier_model": outlier}


def compute_taxonomy_difficulty(predictions: dict) -> dict:
    models = sorted(predictions.keys())
    all_papers = set()
    for m in models:
        all_papers.update(predictions[m].keys())
    all_taxonomies = set()
    for m in models:
        for p in predictions[m]:
            all_taxonomies.update(predictions[m][p].keys())

    tax_stats = {}
    for tax in sorted(all_taxonomies):
        unanimous = majority = total = 0
        confidences = []
        chapters_used: set[str] = set()
        for paper in sorted(all_papers):
            top1s = []
            for m in models:
                chs = predictions.get(m, {}).get(paper, {}).get(tax)
                if chs:
                    top1s.append(chs[0]["chapter"])
                    confidences.append(chs[0]["confidence"])
                    chapters_used.add(chs[0]["chapter"])
            if len(top1s) < 2:
                continue
            total += 1
            counts = Counter(top1s)
            top_count = counts.most_common(1)[0][1]
            if top_count == len(top1s):
                unanimous += 1
            elif top_count > len(top1s) / 2:
                majority += 1

        chapter_counts = Counter()
        for paper in all_papers:
            for m in models:
                chs = predictions.get(m, {}).get(paper, {}).get(tax)
                if chs:
                    chapter_counts[chs[0]["chapter"]] += 1
        total_assignments = sum(chapter_counts.values())
        entropy = 0.0
        if total_assignments > 0:
            for cnt in chapter_counts.values():
                p = cnt / total_assignments
                if p > 0:
                    entropy -= p * math.log2(p)

        tax_stats[tax] = {
            "unanimous_rate": round(unanimous / total, 4) if total else 0,
            "majority_rate": round((unanimous + majority) / total, 4) if total else 0,
            "mean_top1_confidence": round(mean(confidences), 1) if confidences else 0,
            "unique_chapters_used": len(chapters_used),
            "entropy": round(entropy, 3), "n_papers": total,
        }

    ranked = sorted(tax_stats.items(), key=lambda x: x[1]["unanimous_rate"], reverse=True)
    return {"taxonomies": tax_stats, "easiest": ranked[0][0] if ranked else None, "hardest": ranked[-1][0] if ranked else None}


def _collect_leaf_chapters(items: list[dict], result: set | None = None) -> set[str]:
    if result is None:
        result = set()
    for item in items:
        sub = item.get("items", [])
        if sub:
            _collect_leaf_chapters(sub, result)
        else:
            ch_val = item.get("chapter") or item.get("name", "")
            result.add(str(ch_val))
    return result


def _collect_all_chapter_names(items: list[dict], result: dict | None = None) -> dict[str, str]:
    if result is None:
        result = {}
    for item in items:
        ch_val = item.get("chapter") or item.get("name", "")
        result[str(ch_val)] = item.get("name", "")
        if item.get("items"):
            _collect_all_chapter_names(item["items"], result)
    return result


def _gini(counts: list[int]) -> float:
    n = len(counts)
    if n == 0 or sum(counts) == 0:
        return 0.0
    vals = sorted(counts)
    total = sum(vals)
    cumsum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(vals))
    return cumsum / (n * total)


def _normalized_entropy(counts: list[int]) -> float:
    n = len(counts)
    if n <= 1 or sum(counts) == 0:
        return 0.0
    total = sum(counts)
    h = -sum((c / total) * math.log2(c / total) for c in counts if c > 0)
    return h / math.log2(n)


def compute_coverage_analysis(consensus_dir: Path, taxonomies_dir: Path) -> dict:
    tax_leaves: dict[str, set[str]] = {}
    tax_names: dict[str, dict[str, str]] = {}
    for json_path in sorted(taxonomies_dir.glob("*.json")):
        items = json.loads(json_path.read_text(encoding="utf-8"))
        tax_leaves[json_path.stem] = _collect_leaf_chapters(items)
        tax_names[json_path.stem] = _collect_all_chapter_names(items)

    if not tax_leaves:
        return {}

    tax_counts: dict[str, Counter] = defaultdict(Counter)
    n_papers = 0
    for json_path in sorted(consensus_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        n_papers += 1
        for tax_name, tax_result in data.get("classifications", {}).items():
            ch = tax_result.get("chapter")
            if ch:
                tax_counts[tax_name][str(ch)] += 1

    result = {}
    fat_multiplier = 2.5
    for tax_name, leaves in sorted(tax_leaves.items()):
        counts_dict = tax_counts.get(tax_name, Counter())
        leaf_counts = {ch: counts_dict.get(ch, 0) for ch in leaves}
        count_vals = list(leaf_counts.values())
        n_leaves = len(leaves)
        mean_c = mean(count_vals) if count_vals else 0.0
        fat_threshold = mean_c * fat_multiplier
        dead = sorted(ch for ch, c in leaf_counts.items() if c == 0)
        fat = sorted(ch for ch, c in leaf_counts.items() if c > fat_threshold and n_leaves > 1)
        per_leaf = {ch: {"name": tax_names[tax_name].get(ch, ch), "count": leaf_counts[ch], "dead": ch in dead, "fat": ch in fat} for ch in sorted(leaves)}
        result[tax_name] = {
            "n_leaves": n_leaves, "n_papers": n_papers, "mean_papers_per_leaf": round(mean_c, 1),
            "dead_count": len(dead), "fat_count": len(fat),
            "gini": round(_gini(count_vals), 3), "normalized_entropy": round(_normalized_entropy(count_vals), 3),
            "dead_leaves": dead, "fat_leaves": fat, "per_leaf": per_leaf,
        }
    return result


def _cramers_v(table: list[list[int]]) -> float:
    n_rows = len(table)
    n_cols = len(table[0]) if table else 0
    if n_rows < 2 or n_cols < 2:
        return 0.0
    n = sum(sum(row) for row in table)
    if n == 0:
        return 0.0
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[r][c] for r in range(n_rows)) for c in range(n_cols)]
    chi2 = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            expected = (row_sums[r] * col_sums[c]) / n
            if expected > 0:
                chi2 += (table[r][c] - expected) ** 2 / expected
    min_dim = min(n_rows - 1, n_cols - 1)
    if min_dim == 0:
        return 0.0
    return math.sqrt(chi2 / (n * min_dim))


def compute_taxonomy_independence(consensus_dir: Path) -> dict:
    paper_assignments: dict[str, dict[str, str]] = {}
    all_taxonomies: set[str] = set()
    for json_path in sorted(consensus_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        paper_fn = data["filename"]
        paper_assignments[paper_fn] = {}
        for tax_name, tax_result in data.get("classifications", {}).items():
            ch = tax_result.get("chapter")
            if ch:
                paper_assignments[paper_fn][tax_name] = str(ch)
                all_taxonomies.add(tax_name)

    tax_list = sorted(all_taxonomies)
    if len(tax_list) < 2:
        return {"pairs": {}, "matrix": [], "taxonomy_names": tax_list, "redundant_pairs": []}

    pairs: dict[str, dict] = {}
    redundant_pairs: list[str] = []
    matrix = [[0.0] * len(tax_list) for _ in tax_list]
    for i, tax_a in enumerate(tax_list):
        matrix[i][i] = 1.0
        for j, tax_b in enumerate(tax_list):
            if j <= i:
                continue
            chapters_a = sorted({v for p in paper_assignments.values() if (v := p.get(tax_a))})
            chapters_b = sorted({v for p in paper_assignments.values() if (v := p.get(tax_b))})
            a_idx = {ch: k for k, ch in enumerate(chapters_a)}
            b_idx = {ch: k for k, ch in enumerate(chapters_b)}
            table = [[0] * len(chapters_b) for _ in chapters_a]
            for paper, assignments in paper_assignments.items():
                ca = assignments.get(tax_a)
                cb = assignments.get(tax_b)
                if ca in a_idx and cb in b_idx:
                    table[a_idx[ca]][b_idx[cb]] += 1
            v = round(_cramers_v(table), 4)
            key = f"{tax_a} vs {tax_b}"
            pairs[key] = {"cramers_v": v, "redundant": v > 0.7}
            matrix[i][j] = v
            matrix[j][i] = v
            if v > 0.7:
                redundant_pairs.append(key)

    return {"pairs": pairs, "matrix": matrix, "taxonomy_names": tax_list, "redundant_pairs": redundant_pairs}


# ---------------------------------------------------------------------------
# Metric verdicts (pass/fail for each metric)
# ---------------------------------------------------------------------------

def compute_metric_verdicts(metrics: dict, cfg) -> dict:
    """Evaluate each metric against thresholds and return pass/fail verdicts."""
    th = cfg.get("thresholds", {})
    verdicts = {}

    # M1: Fleiss' Kappa — per-taxonomy breakdown
    fk = metrics["fleiss_kappa"]
    min_fk = float(th.get("min_fleiss_kappa", 0.4))
    fk_val = fk["overall"]
    failing_m1 = {t: v for t, v in fk.get("per_taxonomy", {}).items() if v < min_fk}
    verdicts["m1"] = {
        "id": "m1", "name": "Fleiss' Kappa", "group": "Inter-Model Agreement",
        "passed": fk_val >= min_fk,
        "value": round(fk_val, 4), "threshold": min_fk,
        "detail": f"Overall kappa {fk_val:.3f} >= {min_fk}" if fk_val >= min_fk
                  else f"Overall kappa {fk_val:.3f} < {min_fk}",
        "action": None if fk_val >= min_fk
                  else "Clarify category boundaries and descriptions — models interpret categories inconsistently",
        "failing_taxonomies": list(failing_m1.keys()),
    }

    # M2: Pairwise Agreement — per-taxonomy from pair breakdowns
    pa = metrics["pairwise_agreement"]
    min_pa = float(th.get("min_mean_pairwise_agreement", 0.5))
    min_pk = float(th.get("min_pairwise_kappa", 0.2))
    rates = [p["agreement_rate"] for p in pa["pairs"].values()]
    mean_rate = mean(rates) if rates else 0.0
    low_kappa_pairs = [name for name, p in pa["pairs"].items() if p["cohens_kappa"] < min_pk]
    # Find per-taxonomy low agreement
    per_tax_rates: dict[str, list[float]] = defaultdict(list)
    for pair_data in pa["pairs"].values():
        for tax, tax_info in pair_data.get("per_taxonomy", {}).items():
            per_tax_rates[tax].append(tax_info["agreement_rate"])
    failing_m2 = [t for t, rs in per_tax_rates.items() if mean(rs) < min_pa]
    passed_m2 = mean_rate >= min_pa and not low_kappa_pairs
    verdicts["m2"] = {
        "id": "m2", "name": "Pairwise Agreement", "group": "Inter-Model Agreement",
        "passed": passed_m2,
        "value": round(mean_rate, 4), "threshold": min_pa,
        "detail": f"Mean agreement {mean_rate:.1%} >= {min_pa:.0%}" if mean_rate >= min_pa
                  else f"Mean agreement {mean_rate:.1%} < {min_pa:.0%}",
        "action": None if passed_m2
                  else "Sharpen distinctions between overlapping categories"
                       + (f" (low kappa pairs: {', '.join(low_kappa_pairs)})" if low_kappa_pairs else ""),
        "failing_taxonomies": failing_m2,
    }

    # M3: Consensus Strength — per-taxonomy weak fraction
    cs = metrics["consensus_strength"]
    cs_overall = cs["overall"]
    max_weak = float(th.get("max_weak_consensus_fraction", 0.3))
    weak_frac = cs_overall["weak_count"] / cs_overall["n"] if cs_overall.get("n") else 0
    failing_m3 = []
    for tax, ts in cs.get("per_taxonomy", {}).items():
        t_weak = ts.get("weak_count", 0)
        t_n = ts.get("n", 0)
        if t_n > 0 and t_weak / t_n > max_weak:
            failing_m3.append(tax)
    verdicts["m3"] = {
        "id": "m3", "name": "Consensus Strength", "group": "Consensus Quality",
        "passed": weak_frac <= max_weak,
        "value": round(weak_frac, 4), "threshold": max_weak,
        "detail": f"Weak consensus {weak_frac:.0%} of items <= {max_weak:.0%}" if weak_frac <= max_weak
                  else f"Weak consensus {weak_frac:.0%} of items > {max_weak:.0%}",
        "action": None if weak_frac <= max_weak
                  else "Merge or differentiate boundary categories — too many papers split between two choices",
        "failing_taxonomies": failing_m3,
    }

    # M4: Confidence-Agreement Correlation (global, not per-taxonomy)
    ca = metrics["confidence_agreement"]
    min_r = float(th.get("min_confidence_agreement_r", 0.3))
    r_val = ca["pearson_r"]
    verdicts["m4"] = {
        "id": "m4", "name": "Confidence-Agreement Correlation", "group": "Consensus Quality",
        "passed": abs(r_val) >= min_r,
        "value": round(r_val, 4), "threshold": min_r,
        "detail": f"Pearson r = {r_val:.3f} (|r| >= {min_r})" if abs(r_val) >= min_r
                  else f"Pearson r = {r_val:.3f} (|r| < {min_r})",
        "action": None if abs(r_val) >= min_r
                  else "Category descriptions may be misleading — models are confident but disagree",
        "failing_taxonomies": [],  # global metric, applies to all
    }

    # M5: Per-Model Deviation (informational, no auto-fix)
    md = metrics["model_deviation"]
    max_dev = float(th.get("max_outlier_deviation", 0.4))
    max_rate = max((s["deviation_rate"] for s in md["models"].values()), default=0)
    verdicts["m5"] = {
        "id": "m5", "name": "Per-Model Deviation", "group": "Model Outliers",
        "passed": max_rate <= max_dev,
        "value": round(max_rate, 4), "threshold": max_dev,
        "detail": f"Max deviation {max_rate:.1%} <= {max_dev:.0%}" if max_rate <= max_dev
                  else f"Max deviation {max_rate:.1%} > {max_dev:.0%} (outlier: {md['outlier_model']})",
        "action": None if max_rate <= max_dev
                  else "Informational: consider excluding outlier model or investigate its disagreement patterns",
        "failing_taxonomies": [],
    }

    # M6: Taxonomy Difficulty
    td = metrics["taxonomy_difficulty"]["taxonomies"]
    min_unan = float(th.get("min_unanimous_rate", 0.3))
    failing_m6 = [t for t, s in td.items() if s["unanimous_rate"] < min_unan]
    verdicts["m6"] = {
        "id": "m6", "name": "Taxonomy Difficulty", "group": "Taxonomy Clarity",
        "passed": len(failing_m6) == 0,
        "value": round(min(s["unanimous_rate"] for s in td.values()), 4) if td else 0,
        "threshold": min_unan,
        "detail": f"All taxonomies have unanimous rate >= {min_unan:.0%}" if not failing_m6
                  else f"{len(failing_m6)} taxonomy(ies) below {min_unan:.0%} unanimous: {', '.join(failing_m6)}",
        "action": None if not failing_m6
                  else "Taxonomy too fine-grained — consider merging similar leaf categories",
        "failing_taxonomies": failing_m6,
    }

    # M7: Coverage Analysis — per-taxonomy
    cov = metrics.get("coverage_analysis", {})
    max_gini = float(th.get("max_gini", 0.5))
    min_ent = float(cfg.get("min_coverage_entropy", 0.7))
    dead_thresh = int(cfg.get("dead_threshold", 0))
    cov_issues = []
    failing_m7 = []
    for tax, info in cov.items():
        tax_fail = False
        if info["dead_count"] > 0:
            cov_issues.append(f"{tax}: {info['dead_count']} dead leaves")
            tax_fail = True
        if info["gini"] > max_gini:
            cov_issues.append(f"{tax}: gini {info['gini']:.3f} > {max_gini}")
            tax_fail = True
        if info["normalized_entropy"] < min_ent:
            cov_issues.append(f"{tax}: entropy {info['normalized_entropy']:.3f} < {min_ent}")
            tax_fail = True
        if tax_fail:
            failing_m7.append(tax)
    verdicts["m7"] = {
        "id": "m7", "name": "Coverage Analysis", "group": "Structural Balance",
        "passed": len(cov_issues) == 0,
        "value": None,
        "threshold": {"max_gini": max_gini, "min_entropy": min_ent, "dead_threshold": dead_thresh},
        "detail": "All taxonomies structurally balanced" if not cov_issues
                  else "; ".join(cov_issues),
        "action": None if not cov_issues
                  else "Rebalance: split overcrowded leaves, merge/replace empty ones",
        "failing_taxonomies": failing_m7,
    }

    # M8: Taxonomy Independence (cross-taxonomy, not per-taxonomy)
    ti = metrics.get("taxonomy_independence", {})
    max_v = float(th.get("max_cramers_v", 0.7))
    redundant = ti.get("redundant_pairs", [])
    verdicts["m8"] = {
        "id": "m8", "name": "Taxonomy Independence", "group": "Taxonomy Diversity",
        "passed": len(redundant) == 0,
        "value": None,
        "threshold": max_v,
        "detail": "All taxonomy pairs are adequately independent" if not redundant
                  else f"Redundant pairs (V > {max_v}): {', '.join(redundant)}",
        "action": None if not redundant
                  else "Redundant taxonomies detected — consider merging or replacing one",
        "failing_taxonomies": [],  # cross-taxonomy; redundant_pairs has the pair names
        "redundant_pairs": redundant,
    }

    return verdicts


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    meta = metrics["metadata"]
    print(f"\n{'=' * 60}")
    print(f"  Internal Consistency Report")
    print(f"  Topic: {meta['topic']}  |  {meta['n_models']} models  |  {meta['n_papers']} papers")
    print(f"  Models: {', '.join(meta['models'])}")
    print(f"{'=' * 60}")

    fk = metrics["fleiss_kappa"]
    print(f"\n--- Fleiss' Kappa (multi-rater agreement) ---")
    print(f"  Overall: {fk['overall']}")
    for tax, val in sorted(fk["per_taxonomy"].items()):
        print(f"  {tax}: {val}")

    pa = metrics["pairwise_agreement"]
    print(f"\n--- Pairwise Agreement ---")
    for pair_name, pair_data in pa["pairs"].items():
        print(f"  {pair_name}: {pair_data['agreement_rate']:.1%} (kappa={pair_data['cohens_kappa']:.3f}, n={pair_data['n_compared']})")

    cs = metrics["consensus_strength"]["overall"]
    print(f"\n--- Consensus Strength (winning margin) ---")
    print(f"  Mean: {cs['mean']:.1f}  Median: {cs['median']:.1f}  Std: {cs['std']:.1f}")
    print(f"  Weak (<20): {cs['weak_count']}/{cs['n']}  Strong (>=20): {cs['strong_count']}/{cs['n']}")

    ca = metrics["confidence_agreement"]
    print(f"\n--- Confidence-Agreement Correlation ---")
    print(f"  Pearson r: {ca['pearson_r']:.4f}  (n={ca['n_points']})")

    md = metrics["model_deviation"]
    print(f"\n--- Per-Model Deviation from Consensus ---")
    for m, stats in sorted(md["models"].items(), key=lambda x: x[1]["deviation_rate"]):
        print(f"  {m}: {stats['deviation_rate']:.1%} ({stats['deviations']}/{stats['total']})")
    print(f"  Outlier: {md['outlier_model']}")

    td = metrics["taxonomy_difficulty"]
    print(f"\n--- Taxonomy Difficulty ---")
    for tax, stats in sorted(td["taxonomies"].items(), key=lambda x: x[1]["unanimous_rate"], reverse=True):
        print(f"  {tax}: unanimous={stats['unanimous_rate']:.0%} majority={stats['majority_rate']:.0%} "
              f"confidence={stats['mean_top1_confidence']:.0f} entropy={stats['entropy']:.2f}")
    print(f"  Easiest: {td['easiest']}  |  Hardest: {td['hardest']}")

    ca = metrics.get("coverage_analysis", {})
    if ca:
        print(f"\n--- Coverage Analysis (papers per leaf, from consensus) ---")
        for tax, info in sorted(ca.items()):
            status = "balanced" if info["dead_count"] == 0 and info["fat_count"] == 0 else "imbalanced"
            print(f"  {tax}: {status}  leaves={info['n_leaves']}  dead={info['dead_count']}  fat={info['fat_count']}"
                  f"  gini={info['gini']:.3f}  entropy={info['normalized_entropy']:.3f}")
            if info["dead_leaves"]:
                print(f"    dead: {', '.join(info['dead_leaves'])}")
            if info["fat_leaves"]:
                print(f"    fat: {', '.join(info['fat_leaves'])}")

    ti = metrics.get("taxonomy_independence", {})
    if ti and ti.get("pairs"):
        print(f"\n--- Cross-Taxonomy Independence (Cramer's V) ---")
        print(f"  (0 = independent/diverse, 1 = redundant)")
        for pair_name, pair_data in sorted(ti["pairs"].items(), key=lambda x: -x[1]["cramers_v"]):
            flag = "  <- REDUNDANT" if pair_data["redundant"] else ""
            print(f"  {pair_name}: V={pair_data['cramers_v']:.4f}{flag}")
        if ti["redundant_pairs"]:
            print(f"  Redundant pairs (V>0.7): {ti['redundant_pairs']}")
        else:
            print(f"  All taxonomy pairs are adequately independent.")

    # Metric verdicts summary
    verdicts = metrics.get("verdicts", {})
    if verdicts:
        passed = sum(1 for v in verdicts.values() if v["passed"])
        total = len(verdicts)
        print(f"\n{'=' * 60}")
        print(f"  Metric Verdicts: {passed}/{total} pass")
        print(f"{'=' * 60}")
        for mid in sorted(verdicts):
            v = verdicts[mid]
            icon = "PASS" if v["passed"] else "FAIL"
            print(f"  [{icon}] {v['id'].upper()}: {v['name']} — {v['detail']}")
            if v["action"]:
                print(f"         -> {v['action']}")

    print()


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_plots(metrics: dict, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    pa = metrics["pairwise_agreement"]
    names = [n.split("/")[-1] for n in pa["model_names"]]
    matrix = pa["matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im)
    ax.set_title("Pairwise Agreement Rate")
    fig.tight_layout()
    fig.savefig(output_dir / "agreement_heatmap.png", dpi=150)
    plt.close(fig)

    margins = [p[tax]["margin"] for p in metrics["consensus_strength"]["per_paper"].values() for tax in p]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(margins, bins=20, edgecolor="white", color="#4a90d9")
    ax.axvline(20, color="red", linestyle="--", alpha=0.7, label="Weak threshold (20)")
    ax.set_xlabel("Winning Margin")
    ax.set_ylabel("Count")
    ax.set_title("Consensus Strength Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "consensus_strength.png", dpi=150)
    plt.close(fig)

    ca = metrics["confidence_agreement"]
    if ca["points"]:
        agreements = [p["agreement"] for p in ca["points"]]
        confidences = [p["mean_confidence"] for p in ca["points"]]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(agreements, confidences, alpha=0.6, edgecolors="white", s=40)
        ax.set_xlabel("Agreement Level")
        ax.set_ylabel("Mean Top-1 Confidence")
        ax.set_title(f"Confidence vs Agreement (r={ca['pearson_r']:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / "confidence_agreement.png", dpi=150)
        plt.close(fig)

    td = metrics["taxonomy_difficulty"]["taxonomies"]
    tax_names = sorted(td.keys())
    short_names = [t.replace("by_", "") for t in tax_names]
    unanimous = [td[t]["unanimous_rate"] for t in tax_names]
    majority_only = [td[t]["majority_rate"] - td[t]["unanimous_rate"] for t in tax_names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(short_names, unanimous, label="Unanimous", color="#27ae60")
    ax.barh(short_names, majority_only, left=unanimous, label="Majority", color="#f39c12")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_title("Taxonomy Difficulty (Agreement Rates)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "taxonomy_difficulty.png", dpi=150)
    plt.close(fig)

    md = metrics["model_deviation"]["models"]
    model_names = sorted(md.keys())
    short_model = [n.split("/")[-1] for n in model_names]
    dev_rates = [md[m]["deviation_rate"] for m in model_names]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if m == metrics["model_deviation"]["outlier_model"] else "#3498db" for m in model_names]
    ax.barh(short_model, dev_rates, color=colors)
    ax.set_xlim(0, max(dev_rates) * 1.3 if dev_rates else 1)
    ax.set_xlabel("Deviation Rate from Consensus")
    ax.set_title("Per-Model Deviation (red = outlier)")
    fig.tight_layout()
    fig.savefig(output_dir / "model_deviation.png", dpi=150)
    plt.close(fig)

    ti = metrics.get("taxonomy_independence", {})
    if ti and ti.get("matrix") and len(ti["taxonomy_names"]) >= 2:
        names = [t.replace("by_", "") for t in ti["taxonomy_names"]]
        mat = ti["matrix"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im)
        ax.set_title("Cross-Taxonomy Independence (Cramer's V)\n0=independent, 1=redundant")
        fig.tight_layout()
        fig.savefig(output_dir / "taxonomy_independence.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Refinement — verdict-driven taxonomy improvement
# ---------------------------------------------------------------------------

# System prompts per refinement type
REFINE_SYSTEM_BALANCE = (
    "You are an expert research taxonomist. You will be given an existing taxonomy "
    "and coverage data showing how papers distributed across its categories. "
    "Restructure the taxonomy to fix balance issues while preserving the organizing principle. "
    "Respond with valid JSON only."
)

REFINE_SYSTEM_BOUNDARIES = (
    "You are an expert research taxonomist. You will be given an existing taxonomy "
    "and evidence that classification models disagree on category boundaries. "
    "Your task is to rewrite category descriptions, phrases, and hard_negatives to make "
    "boundaries unambiguous — without changing the overall structure. "
    "Respond with valid JSON only."
)

REFINE_SYSTEM_REDUNDANCY = (
    "You are an expert research taxonomist. You will be given two taxonomies that are "
    "statistically redundant (they classify papers nearly identically). "
    "Your task is to redesign one of them so it provides a genuinely different organizing perspective. "
    "Respond with valid JSON only."
)

TAXONOMY_SCHEMA_EXAMPLE = """\
{
  "chapter": "1",
  "name": "top_level_category",
  "description": "2-3 sentence description of what this category covers.",
  "items": [
    {
      "name": "subcategory_name",
      "chapter": "1.1",
      "description": "2-3 sentence description.",
      "phrases": ["key phrase 1", "key phrase 2"],
      "hard_negative": "What does NOT belong here and why."
    }
  ]
}"""


def slugify_model(model: str) -> str:
    return model.split("/")[-1]


def _prompt_topic_selection(papers_base: Path) -> Path:
    available = sorted(
        d for d in papers_base.iterdir()
        if d.is_dir() and (d / "classification").is_dir()
    )
    if not available:
        return papers_base
    print("Available topics:")
    for i, d in enumerate(available):
        print(f"  [{i + 1}] {d.name}")
    raw = input(f"Select topic [1-{len(available)}]: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(available):
        return available[int(raw) - 1]
    return available[0]


def parse_json_response(text: str) -> list | dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


async def fetch_all_pricing(base_url: str, api_key: str) -> dict[str, tuple[float, float]]:
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


async def call_llm(
    client: AsyncOpenAI, system: str, user: str,
    model: str, temperature: float, max_tokens: int, token_usage: dict,
) -> list | dict:
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


def _load_consensus_coverage(consensus_dir: Path, taxonomy_name: str) -> dict[str, int]:
    counts: Counter = Counter()
    for json_path in sorted(consensus_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        chapter = data.get("classifications", {}).get(taxonomy_name, {}).get("chapter")
        if chapter:
            counts[str(chapter)] += 1
    return dict(counts)


def _compute_refine_stats(
    coverage: dict[str, int], leaf_chapters: set[str], all_chapters: dict[str, str],
    total_papers: int, dead_threshold: int, fat_multiplier: float,
) -> dict:
    leaf_counts = {ch: coverage.get(ch, 0) for ch in leaf_chapters}
    counts = list(leaf_counts.values())
    n_leaves = len(leaf_chapters)
    mean_count = mean(counts) if counts else 0.0
    fat_threshold = mean_count * fat_multiplier
    dead = [ch for ch, c in leaf_counts.items() if c <= dead_threshold]
    fat = [ch for ch, c in leaf_counts.items() if c > fat_threshold and n_leaves > 1]
    per_leaf = {ch: {"name": all_chapters.get(ch, ch), "count": leaf_counts[ch], "dead": ch in dead, "fat": ch in fat} for ch in sorted(leaf_chapters)}
    return {
        "n_leaves": n_leaves, "total_papers": total_papers,
        "mean_papers_per_leaf": round(mean_count, 1),
        "dead_count": len(dead), "fat_count": len(fat),
        "gini": round(_gini(counts), 3), "normalized_entropy": round(_normalized_entropy(counts), 3),
        "per_leaf": per_leaf, "dead_leaves": sorted(dead), "fat_leaves": sorted(fat),
    }


def _build_coverage_refine_prompt(taxonomy_name: str, taxonomy_json: list, stats: dict) -> str:
    n = stats["total_papers"]
    mean_c = stats["mean_papers_per_leaf"]
    lines = []
    for ch, info in stats["per_leaf"].items():
        flags = []
        if info["dead"]:
            flags.append("DEAD")
        if info["fat"]:
            flags.append(f"FAT ({info['count'] / n * 100:.0f}% of papers)")
        flag_str = "  <- " + ", ".join(flags) if flags else ""
        lines.append(f"  {ch:8}  {info['name']:40}  {info['count']:3} papers{flag_str}")
    coverage_table = "\n".join(lines)

    issues = []
    if stats["dead_count"] > 0:
        dead_names = [f"{ch} ({stats['per_leaf'][ch]['name']})" for ch in stats["dead_leaves"]]
        issues.append(f"  - Dead categories (0 papers): {', '.join(dead_names)}")
    if stats["fat_count"] > 0:
        fat_names = [f"{ch} ({stats['per_leaf'][ch]['name']})" for ch in stats["fat_leaves"]]
        issues.append(f"  - Overcrowded categories: {', '.join(fat_names)}")
    issues_text = "\n".join(issues) if issues else "  None detected (refinement triggered by low entropy)."

    restructure = []
    for ch in stats["fat_leaves"]:
        name = stats["per_leaf"][ch]["name"]
        restructure.append(f"  - Split '{name}' ({ch}) into 2-3 more specific subcategories")
    for ch in stats["dead_leaves"]:
        name = stats["per_leaf"][ch]["name"]
        restructure.append(f"  - Replace or merge '{name}' ({ch}) — it matched no papers")
    if not restructure:
        restructure.append("  - Redistribute categories more evenly")
    restructure_text = "\n".join(restructure)

    return f"""You are an expert research taxonomist. Below is the taxonomy "{taxonomy_name}" and how {n} papers distributed across its leaf categories.

## Current Taxonomy (JSON)
```json
{json.dumps(taxonomy_json, indent=2)}
```

## Coverage Report ({n} papers, {stats['n_leaves']} leaf categories, mean {stats['mean_papers_per_leaf']:.1f} papers/leaf)
{coverage_table}

Balance metrics: Gini={stats['gini']:.3f}, Normalized Entropy={stats['normalized_entropy']:.3f}

## Issues Detected
{issues_text}

## Restructuring Instructions
{restructure_text}

## Requirements
- Keep the same top-level organizing principle ("{taxonomy_name}")
- Maintain 2-3 levels of hierarchy, targeting 5-15 leaf categories total
- Use snake_case for all category names
- Do NOT assign papers to categories
- Every leaf must have: name, chapter, description, phrases, hard_negative
- Chapter numbering: "1", "1.1", "1.1.1", "1.1(a)"

Single category schema example:
{TAXONOMY_SCHEMA_EXAMPLE}

Output a single JSON object:
{{
  "perspective": "{taxonomy_name}",
  "description": "1 sentence explaining this organizing angle.",
  "categories": [ ... ]
}}"""


_TAXONOMY_OUTPUT_INSTRUCTIONS = """\
## Requirements
- Keep the same top-level organizing principle ("{taxonomy_name}")
- Maintain 2-3 levels of hierarchy, targeting 5-15 leaf categories total
- Use snake_case for all category names
- Do NOT assign papers to categories
- Every leaf must have: name, chapter, description, phrases, hard_negative
- Chapter numbering: "1", "1.1", "1.1.1", "1.1(a)"

Single category schema example:
{schema_example}

Output a single JSON object:
{{
  "perspective": "{taxonomy_name}",
  "description": "1 sentence explaining this organizing angle.",
  "categories": [ ... ]
}}"""


def _build_boundary_refine_prompt(
    taxonomy_name: str,
    taxonomy_json: list,
    failing_metrics: list[str],
    metrics: dict,
) -> str:
    """Build prompt for boundary clarification (M1, M2, M3, M4, M6 failures)."""
    # Gather per-taxonomy agreement data
    evidence_lines = []

    fk = metrics.get("fleiss_kappa", {})
    fk_tax = fk.get("per_taxonomy", {}).get(taxonomy_name)
    if fk_tax is not None:
        evidence_lines.append(f"  - Fleiss' kappa for this taxonomy: {fk_tax:.3f}")

    pa = metrics.get("pairwise_agreement", {})
    for pair_name, pair_data in pa.get("pairs", {}).items():
        tax_info = pair_data.get("per_taxonomy", {}).get(taxonomy_name)
        if tax_info:
            evidence_lines.append(
                f"  - {pair_name}: agreement={tax_info['agreement_rate']:.1%}, "
                f"kappa={tax_info['cohens_kappa']:.3f} (n={tax_info['n']})"
            )

    cs = metrics.get("consensus_strength", {}).get("per_taxonomy", {}).get(taxonomy_name)
    if cs:
        evidence_lines.append(
            f"  - Consensus margin: mean={cs.get('mean', 0):.1f}, "
            f"weak={cs.get('weak_count', 0)}/{cs.get('n', 0)}"
        )

    td = metrics.get("taxonomy_difficulty", {}).get("taxonomies", {}).get(taxonomy_name)
    if td:
        evidence_lines.append(
            f"  - Difficulty: unanimous={td['unanimous_rate']:.0%}, "
            f"majority={td['majority_rate']:.0%}, entropy={td['entropy']:.2f}"
        )

    evidence_text = "\n".join(evidence_lines) if evidence_lines else "  (no per-taxonomy data)"

    instructions = []
    if "m1" in failing_metrics or "m2" in failing_metrics:
        instructions.append(
            "- Rewrite descriptions to be more precise and mutually exclusive")
        instructions.append(
            "- Add or improve hard_negative fields to explicitly state what does NOT belong")
        instructions.append(
            "- Expand phrases lists with discriminating keywords")
    if "m3" in failing_metrics:
        instructions.append(
            "- Categories that papers frequently split between should be merged or given sharper boundaries")
    if "m4" in failing_metrics:
        instructions.append(
            "- Descriptions may be misleading models into false confidence — rewrite for clarity")
    if "m6" in failing_metrics:
        instructions.append(
            "- Consider merging similar leaf categories to reduce granularity where models cannot agree")
    instructions_text = "\n".join(instructions)

    output_inst = _TAXONOMY_OUTPUT_INSTRUCTIONS.format(
        taxonomy_name=taxonomy_name, schema_example=TAXONOMY_SCHEMA_EXAMPLE)

    return f"""The taxonomy "{taxonomy_name}" has boundary clarity issues — classification models disagree on where papers belong.

## Current Taxonomy (JSON)
```json
{json.dumps(taxonomy_json, indent=2)}
```

## Agreement Evidence
Failing metrics: {', '.join(m.upper() for m in failing_metrics)}
{evidence_text}

## Instructions
Improve the taxonomy so that category boundaries are unambiguous. You may restructure if needed, but prioritize clarity over structural change.

{instructions_text}

{output_inst}"""


def _build_redundancy_refine_prompt(
    taxonomy_a_name: str, taxonomy_a_json: list,
    taxonomy_b_name: str, taxonomy_b_json: list,
    cramers_v: float,
) -> str:
    """Build prompt for taxonomy redundancy resolution (M8 failure)."""
    return f"""Two taxonomies classify papers nearly identically (Cramer's V = {cramers_v:.3f}, where >0.7 indicates redundancy).

## Taxonomy A: "{taxonomy_a_name}"
```json
{json.dumps(taxonomy_a_json, indent=2)}
```

## Taxonomy B: "{taxonomy_b_name}"
```json
{json.dumps(taxonomy_b_json, indent=2)}
```

## Instructions
Redesign taxonomy B ("{taxonomy_b_name}") so it provides a genuinely different organizing perspective from taxonomy A. The new taxonomy should:
- Capture a different dimension of the research (e.g., if A is by method, B should be by application or by scale)
- Be orthogonal to A: a paper's category in A should NOT predict its category in B
- Cover the same set of papers but from a fresh angle

## Requirements
- Maintain 2-3 levels of hierarchy, targeting 5-15 leaf categories total
- Use snake_case for all category names
- Do NOT assign papers to categories
- Every leaf must have: name, chapter, description, phrases, hard_negative
- Chapter numbering: "1", "1.1", "1.1.1", "1.1(a)"

Single category schema example:
{TAXONOMY_SCHEMA_EXAMPLE}

Output a single JSON object:
{{
  "perspective": "{taxonomy_b_name}",
  "description": "1 sentence explaining this NEW organizing angle.",
  "categories": [ ... ]
}}"""


def _collect_per_taxonomy_failures(verdicts: dict, all_taxonomy_names: list[str]) -> dict[str, list[str]]:
    """Collect which metrics failed for each taxonomy. Returns {tax_name: [metric_ids]}."""
    boundary_metrics = ["m1", "m2", "m3", "m6"]

    tax_failures: dict[str, list[str]] = defaultdict(list)

    for mid in boundary_metrics:
        v = verdicts.get(mid)
        if not v or v["passed"]:
            continue
        for tax in v.get("failing_taxonomies", []):
            tax_failures[tax].append(mid)

    # M4 is global — if it fails, apply to taxonomies with other boundary failures,
    # or to ALL taxonomies if no other boundary metric failed
    m4 = verdicts.get("m4")
    if m4 and not m4["passed"]:
        target_taxes = set(tax_failures.keys())
        if not target_taxes:
            target_taxes = set(all_taxonomy_names)
        for tax in target_taxes:
            if "m4" not in tax_failures[tax]:
                tax_failures[tax].append("m4")

    return dict(tax_failures)


async def _refine_one_taxonomy(
    client: AsyncOpenAI, ref, pricing_map: dict,
    tax_path: Path, system_prompt: str, user_prompt: str, label: str,
) -> dict | None:
    """Call LLM to refine a taxonomy and save the result. Returns summary dict or None."""
    # Snapshot old leaves
    old_json = json.loads(tax_path.read_text(encoding="utf-8"))
    old_leaves = _collect_leaf_chapters(old_json if isinstance(old_json, list) else old_json.get("categories", old_json))
    old_names = _collect_all_chapter_names(old_json if isinstance(old_json, list) else old_json.get("categories", old_json))

    token_usage = {"prompt": 0, "completion": 0}
    result = await call_llm(
        client, system_prompt, user_prompt,
        ref.refiner_model, float(ref.refiner_temperature), int(ref.max_tokens), token_usage,
    )

    if isinstance(result, dict):
        categories = result.get("categories", [])
    elif isinstance(result, list):
        categories = result
    else:
        log.error("Unexpected refiner response type for %s: %s", label, type(result))
        return None

    if not categories:
        log.error("Refiner returned empty categories for %s", label)
        return None

    tax_path.write_text(json.dumps(categories, indent=2, ensure_ascii=False), encoding="utf-8")
    print_cost(slugify_model(ref.refiner_model), token_usage, pricing_map.get(ref.refiner_model))
    print(f"    -> saved refined taxonomy")

    # Compute diff
    new_leaves = _collect_leaf_chapters(categories)
    new_names = _collect_all_chapter_names(categories)
    # Compare by name since chapter numbers may change
    old_name_set = set(old_names.values()) if old_names else old_leaves
    new_name_set = set(new_names.values()) if new_names else new_leaves
    added = sorted(new_name_set - old_name_set)
    removed = sorted(old_name_set - new_name_set)
    kept = len(old_name_set & new_name_set)

    return {
        "taxonomy": tax_path.stem,
        "phase": label.split(":")[0],
        "old_leaves": len(old_leaves),
        "new_leaves": len(new_leaves),
        "added": added,
        "removed": removed,
        "kept": kept,
    }


async def run_refinement(cfg: DictConfig, topic_dir: Path, metrics: dict) -> bool:
    """Run verdict-driven taxonomy refinement. Returns True if any taxonomy was refined."""
    taxonomies_dir = topic_dir / "taxonomies"
    consensus_dir = topic_dir / "classification" / "_consensus"

    if not taxonomies_dir.exists() or not consensus_dir.exists():
        log.error("Missing taxonomies or consensus directory")
        return False

    taxonomy_files = {f.stem: f for f in sorted(taxonomies_dir.glob("by_*.json")) if f.is_file()}
    consensus_files = list(consensus_dir.glob("*.json"))
    total_papers = len(consensus_files)

    if not taxonomy_files or total_papers == 0:
        log.error("No taxonomy files or consensus files found")
        return False

    ref = cfg.refinement
    verdicts = metrics.get("verdicts", {})
    print(f"\n--- Verdict-Driven Refinement ---")
    print(f"  Refiner: {slugify_model(ref.refiner_model)}")

    # Summarize which metrics failed
    failed = {mid: v for mid, v in verdicts.items() if not v["passed"]}
    if not failed:
        print(f"  All metrics pass — no refinement needed.")
        return False

    print(f"  Failed metrics: {', '.join(v['id'].upper() + ' (' + v['name'] + ')' for v in failed.values())}")

    client = AsyncOpenAI(api_key=ref.api_key, base_url=ref.base_url)
    pricing_map = await fetch_all_pricing(ref.base_url, ref.api_key)
    summaries: list[dict] = []

    # --- Phase 1: M7 structural balance (existing logic) ---
    m7 = verdicts.get("m7")
    if m7 and not m7["passed"]:
        print(f"\n  Phase 1: Structural balance (M7)")
        for tax_name in m7.get("failing_taxonomies", []):
            tax_path = taxonomy_files.get(tax_name)
            if not tax_path:
                continue
            taxonomy_json = json.loads(tax_path.read_text(encoding="utf-8"))
            leaf_chapters = _collect_leaf_chapters(taxonomy_json)
            all_chapters = _collect_all_chapter_names(taxonomy_json)
            coverage = _load_consensus_coverage(consensus_dir, tax_name)

            stats = _compute_refine_stats(
                coverage, leaf_chapters, all_chapters, total_papers,
                dead_threshold=ref.dead_threshold, fat_multiplier=ref.fat_multiplier,
            )

            print(f"\n  [{tax_name}] M7: dead={stats['dead_count']}, fat={stats['fat_count']}, "
                  f"gini={stats['gini']:.3f}, entropy={stats['normalized_entropy']:.3f}")
            print(f"    -> structural rebalancing")

            prompt = _build_coverage_refine_prompt(tax_name, taxonomy_json, stats)
            s = await _refine_one_taxonomy(
                client, ref, pricing_map, tax_path,
                REFINE_SYSTEM_BALANCE, prompt, f"M7:{tax_name}",
            )
            if s:
                summaries.append(s)

    # --- Phase 2: Boundary clarification (M1, M2, M3, M4, M6) ---
    boundary_failures = _collect_per_taxonomy_failures(verdicts, list(taxonomy_files.keys()))
    if boundary_failures:
        print(f"\n  Phase 2: Boundary clarification (M1-M4, M6)")
        for tax_name, failing_metrics in sorted(boundary_failures.items()):
            tax_path = taxonomy_files.get(tax_name)
            if not tax_path:
                continue
            taxonomy_json = json.loads(tax_path.read_text(encoding="utf-8"))

            print(f"\n  [{tax_name}] failing: {', '.join(m.upper() for m in failing_metrics)}")
            print(f"    -> boundary clarification")

            prompt = _build_boundary_refine_prompt(
                tax_name, taxonomy_json, failing_metrics, metrics)
            s = await _refine_one_taxonomy(
                client, ref, pricing_map, tax_path,
                REFINE_SYSTEM_BOUNDARIES, prompt, f"boundary:{tax_name}",
            )
            if s:
                summaries.append(s)

    # --- Phase 3: Redundancy resolution (M8) ---
    m8 = verdicts.get("m8")
    if m8 and not m8["passed"]:
        redundant_pairs = m8.get("redundant_pairs", [])
        ti = metrics.get("taxonomy_independence", {})
        if redundant_pairs:
            print(f"\n  Phase 3: Redundancy resolution (M8)")
        for pair_str in redundant_pairs:
            # pair_str format: "tax_a vs tax_b"
            parts = pair_str.split(" vs ")
            if len(parts) != 2:
                continue
            tax_a_name, tax_b_name = parts[0].strip(), parts[1].strip()
            tax_a_path = taxonomy_files.get(tax_a_name)
            tax_b_path = taxonomy_files.get(tax_b_name)
            if not tax_a_path or not tax_b_path:
                continue
            tax_a_json = json.loads(tax_a_path.read_text(encoding="utf-8"))
            tax_b_json = json.loads(tax_b_path.read_text(encoding="utf-8"))

            v_score = ti.get("pairs", {}).get(pair_str, {}).get("cramers_v", 0.0)
            print(f"\n  [{pair_str}] Cramer's V = {v_score:.3f}")
            print(f"    -> redesigning {tax_b_name}")

            prompt = _build_redundancy_refine_prompt(
                tax_a_name, tax_a_json, tax_b_name, tax_b_json, v_score)
            s = await _refine_one_taxonomy(
                client, ref, pricing_map, tax_b_path,
                REFINE_SYSTEM_REDUNDANCY, prompt, f"M8:{tax_b_name}",
            )
            if s:
                summaries.append(s)

    # --- Phase 4: M5 — informational only ---
    m5 = verdicts.get("m5")
    if m5 and not m5["passed"]:
        print(f"\n  Note (M5): {m5['detail']}")
        print(f"    No automatic fix — consider excluding the outlier model from consensus.")

    # --- Refinement summary ---
    if summaries:
        print(f"\n  {'=' * 60}")
        print(f"  Refinement Summary")
        print(f"  {'=' * 60}")
        print(f"  {'Taxonomy':<32} {'Phase':<12} {'Leaves':<10} {'Added':<8} {'Removed':<8} {'Kept':<6}")
        print(f"  {'-' * 32} {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 6}")
        for s in summaries:
            leaves_str = f"{s['old_leaves']}->{s['new_leaves']}"
            print(f"  {s['taxonomy']:<32} {s['phase']:<12} {leaves_str:<10} "
                  f"+{len(s['added']):<7} -{len(s['removed']):<7} {s['kept']:<6}")
            if s["added"]:
                print(f"    + {', '.join(s['added'][:5])}{'...' if len(s['added']) > 5 else ''}")
            if s["removed"]:
                print(f"    - {', '.join(s['removed'][:5])}{'...' if len(s['removed']) > 5 else ''}")
        print()

    return len(summaries) > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    papers_base = Path(to_absolute_path("papers"))
    topic_dir = _prompt_topic_selection(papers_base)

    classification_dir = topic_dir / "classification"
    taxonomies_dir = topic_dir / "taxonomies"
    output_dir = topic_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    consensus_dir = classification_dir / "_consensus"
    if not consensus_dir.exists():
        log.error("No consensus directory at %s — run 3_classify.py first", consensus_dir)
        return

    # Load model data
    model_dirs = find_model_dirs(classification_dir)
    model_dirs = [d for d in model_dirs if "_consensus" not in d.parts]
    if not model_dirs:
        log.error("No model directories found under %s", classification_dir)
        return

    all_model_results = {}
    for model_dir in model_dirs:
        model_name = str(model_dir.relative_to(classification_dir))
        all_model_results[model_name] = load_classifications(model_dir)

    taxonomy_lookup = load_taxonomy_lookup(taxonomies_dir) if taxonomies_dir.exists() else {}
    predictions = extract_normalized_predictions(all_model_results, taxonomy_lookup)

    # Compute all metrics
    metrics = {
        "pairwise_agreement": compute_pairwise_agreement(predictions),
        "fleiss_kappa": compute_fleiss_kappa(predictions),
        "consensus_strength": compute_consensus_strength(consensus_dir),
        "confidence_agreement": compute_confidence_agreement(predictions),
        "model_deviation": compute_model_deviation(predictions, consensus_dir),
        "taxonomy_difficulty": compute_taxonomy_difficulty(predictions),
        "metadata": {
            "topic": topic_dir.name,
            "n_models": len(all_model_results),
            "n_papers": len(set().union(*(r.keys() for r in all_model_results.values()))),
            "models": sorted(all_model_results.keys()),
            "taxonomies": sorted(taxonomy_lookup.keys()) if taxonomy_lookup else [],
        },
    }

    if taxonomies_dir.exists():
        metrics["coverage_analysis"] = compute_coverage_analysis(consensus_dir, taxonomies_dir)
        metrics["taxonomy_independence"] = compute_taxonomy_independence(consensus_dir)

    metrics["verdicts"] = compute_metric_verdicts(metrics, cfg.refinement)

    print_report(metrics)

    out_path = output_dir / "consistency_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metrics saved to: {out_path}")

    generate_plots(metrics, output_dir / "plots")

    # --- Human gate: Accept or refine? ---
    failed_count = sum(1 for v in metrics.get("verdicts", {}).values() if not v["passed"])
    print(f"\n{'=' * 60}")
    print(f"  Accept results or refine based on failed metrics?")
    print(f"  {failed_count} metric(s) need attention" if failed_count else "  All metrics pass")
    print(f"  [a] Accept — done, proceed to 5_evolve.py")
    print(f"  [r] Refine — auto-fix via boundary/balance/redundancy prompts")
    print(f"{'=' * 60}")

    choice = input("> ").strip().lower()
    if choice != "r":
        print("Accepted. Run: python 5_evolve.py")
        return

    # Run refinement
    any_refined = asyncio.run(run_refinement(cfg, topic_dir, metrics))
    if any_refined:
        print(f"\nTaxonomies refined. Review them in the Web UI /editor.")
        print(f"Then reclassify: python 3_classify.py taxonomy.reclassify=true")
    else:
        print(f"\nAll taxonomies already balanced. No changes made.")


if __name__ == "__main__":
    main()
