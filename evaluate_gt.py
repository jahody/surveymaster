"""Evaluate classification results against Ground Truth (GT)."""

import json
import logging
import random
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import _resolvers  # noqa: F401 – registers OmegaConf resolvers
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def collect_gt_mapping(items: list[dict], mapping: dict[str, str], current_chapter: str = "") -> None:
    """Recursively collect arxiv IDs to ground truth chapter mappings."""
    for item in items:
        chapter = item.get("chapter", str(item.get("name", "")))
        if "papers" in item:
            for paper in item["papers"]:
                if "arxiv" in paper:
                    mapping[paper["arxiv"]] = str(chapter)
        if "items" in item:
            collect_gt_mapping(item["items"], mapping)

def load_gt(topic_dir: Path) -> dict[str, str]:
    """Load all GT taxonomies and return a mapping of arxiv_id -> gt_chapter."""
    gt_dir = topic_dir / "taxonomies_GT"
    if not gt_dir.exists():
        log.warning(f"No taxonomies_GT directory found at {gt_dir}")
        return {}
    
    mapping = {}
    for json_path in gt_dir.glob("*.json"):
        try:
            items = json.loads(json_path.read_text(encoding="utf-8"))
            collect_gt_mapping(items, mapping)
        except Exception as e:
            log.error(f"Error loading GT from {json_path}: {e}")
    return mapping

def load_predictions(output_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Returns {model_name: {paper_arxiv: [{"chapter": "...", "confidence": 0.8}, ...]}}."""
    predictions = {}
    if not output_dir.exists():
        return predictions

    for json_path in output_dir.rglob("*.json"):
        if "eval_plots" in json_path.parts:
            continue
            
        rel_path = json_path.parent.relative_to(output_dir)
        model_name = str(rel_path).replace("\\", "/")
        
        if model_name not in predictions:
            predictions[model_name] = {}

        stem = json_path.stem.split("v")[0] if "v" in json_path.stem else json_path.stem
        
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        
        for tax_name, tax_result in data.get("classifications", {}).items():
            if "error" in tax_result:
                continue
            chapters = tax_result.get("chapters", [])
            if not chapters and "chapter" in tax_result:
                 ch = tax_result["chapter"]
                 if isinstance(ch, list):
                     chapters = [{"chapter": c, "confidence": 100//len(ch)} for c in ch]
                 else:
                     chapters = [{"chapter": ch, "confidence": 100}]
            
            # Normalize confidence to 0.0 - 1.0
            normalized_chapters = []
            for c in chapters:
                conf = float(c.get("confidence", 0)) / 100.0
                normalized_chapters.append({"chapter": str(c.get("chapter", "")), "confidence": conf})
            
            # Sort by confidence
            normalized_chapters.sort(key=lambda x: x["confidence"], reverse=True)
            predictions[model_name][stem] = normalized_chapters
            break # Just evaluate against the first (and usually only) taxonomy
    return predictions

def bootstrap_ci(y_true: list[str], y_pred: list[str], n_iterations: int = 10000) -> tuple[float, float]:
    """Calculate 95% Bootstrap Confidence Interval for accuracy."""
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    
    corrects = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    indices = np.random.randint(0, n, size=(n_iterations, n))
    bootstrap_means = corrects[indices].mean(axis=1)
    
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    return float(lower), float(upper)

def paired_bootstrap_test(y_true: list[str], y_pred_a: list[str], y_pred_b: list[str], n_iterations: int = 10000) -> float:
    """Compare model A (e.g. consensus) against model B. Returns p-value (H0: A <= B)."""
    n = len(y_true)
    if n == 0:
        return 1.0
        
    correct_a = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred_a)])
    correct_b = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred_b)])
    
    diffs = correct_a - correct_b
    observed_diff = diffs.mean()
    if observed_diff <= 0:
        return 1.0 
        
    shifted_diffs = diffs - observed_diff
    indices = np.random.randint(0, n, size=(n_iterations, n))
    bootstrap_diffs = shifted_diffs[indices].mean(axis=1)
    
    p_value = np.sum(bootstrap_diffs >= observed_diff) / n_iterations
    return float(p_value)

def compute_calibration(y_true: list[str], y_pred: list[str], confidences: list[float], n_bins: int = 10) -> tuple[float, float, dict]:
    """Calculate ECE, Brier Score, and return bin statistics for plotting."""
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, {}
        
    corrects = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    confs = np.array(confidences)
    
    brier = float(np.mean((confs - corrects) ** 2))
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confs, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    bin_stats = {"conf": [], "acc": [], "count": [], "edges": bins}
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        if count > 0:
            bin_acc = corrects[mask].mean()
            bin_conf = confs[mask].mean()
            ece += (count / n) * abs(bin_acc - bin_conf)
            
            bin_stats["conf"].append(bin_conf)
            bin_stats["acc"].append(bin_acc)
            bin_stats["count"].append(count)
        else:
            bin_stats["conf"].append(0)
            bin_stats["acc"].append(0)
            bin_stats["count"].append(0)
            
    return float(ece), brier, bin_stats

def plot_reliability_diagram(bin_stats: dict, method_name: str, save_path: Path):
    """Plot separate reliability diagram for a method."""
    if not bin_stats:
        return
        
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    
    centers = (bin_stats["edges"][:-1] + bin_stats["edges"][1:]) / 2
    accs = np.array(bin_stats["acc"])
    counts = np.array(bin_stats["count"])
    
    valid = counts > 0
    if np.any(valid):
        plt.bar(centers[valid], accs[valid], width=0.1, alpha=0.5, edgecolor="black", label="Accuracy")
        plt.plot(centers[valid], accs[valid], 'o-', color="red", label="Reliability Curve")
        
        for c, a, count in zip(centers[valid], accs[valid], counts[valid]):
            plt.text(c, a + 0.02, str(count), ha='center', va='bottom', fontsize=8)
            
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram: {method_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def main(cfg: DictConfig) -> None:
    papers_dir = Path(to_absolute_path(cfg.taxonomy.papers_dir))
    topic_dir = papers_dir.parent
    
    gt_mapping = load_gt(topic_dir)
    if not gt_mapping:
        print(f"No ground truth mapping found in {topic_dir / 'taxonomies_GT'}. Exiting.")
        return
        
    print(f"Loaded {len(gt_mapping)} Ground Truth labels.")
    
    # Scan all classification* directories at the topic level
    sweep_dirs = []
    for d in sorted(topic_dir.iterdir()):
        if d.is_dir() and d.name.startswith("classification"):
            if (d / "_consensus").exists():
                sweep_dirs.append(d)
            else:
                # Check subdirectories for _consensus
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and (sub / "_consensus").exists():
                        sweep_dirs.append(sub)

    if not sweep_dirs:
        print(f"No classification results with _consensus found in {topic_dir}")
        return

    top_k = cfg.taxonomy.get("top_k", 3)
    np.random.seed(42) # For reproducible bootstraps
    
    for s_dir in sweep_dirs:
        print(f"\n{'='*80}")
        print(f"--- Evaluating Sweep: {s_dir.name} ---")
        preds = load_predictions(s_dir)
        
        # Compute Unweighted Majority
        single_models = [m for m in preds.keys() if m != "_consensus"]
        if len(single_models) > 1:
            unweighted_preds = {}
            for arxiv_id in gt_mapping.keys():
                votes = {}
                for m in single_models:
                    if arxiv_id in preds[m] and preds[m][arxiv_id]:
                        vote = preds[m][arxiv_id][0]["chapter"]
                        votes[vote] = votes.get(vote, 0) + 1
                if votes:
                    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
                    winner = sorted_votes[0][0]
                    # Proxy confidence: fraction of models that voted for the winner
                    conf = sorted_votes[0][1] / len(single_models)
                    unweighted_preds[arxiv_id] = [{"chapter": winner, "confidence": conf}]
            preds["Unweighted Majority"] = unweighted_preds

            # Compute Plurality + Confidence Tiebreaker
            plurality_preds = {}
            for arxiv_id in gt_mapping.keys():
                votes = {}
                conf_scores = {}  # chapter -> sum of top-1 confidence
                for m in single_models:
                    if arxiv_id in preds[m] and preds[m][arxiv_id]:
                        vote = preds[m][arxiv_id][0]["chapter"]
                        vote_conf = preds[m][arxiv_id][0]["confidence"]
                        votes[vote] = votes.get(vote, 0) + 1
                        conf_scores[vote] = conf_scores.get(vote, 0) + vote_conf
                if votes:
                    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
                    top_count = sorted_votes[0][1]
                    # Check for majority (>50% of models agree)
                    if top_count > len(single_models) / 2:
                        winner = sorted_votes[0][0]
                        conf = top_count / len(single_models)
                    else:
                        # Tie: fall back to confidence-weighted tiebreaker
                        sorted_by_conf = sorted(conf_scores.items(), key=lambda x: x[1], reverse=True)
                        winner = sorted_by_conf[0][0]
                        total_conf = sum(conf_scores.values())
                        conf = sorted_by_conf[0][1] / total_conf if total_conf > 0 else 0.5
                    plurality_preds[arxiv_id] = [{"chapter": winner, "confidence": conf}]
            preds["Plurality+Tiebreaker"] = plurality_preds

        metrics = {}
        predictions_for_tests = {}
        
        # 1. Compute accuracy and gather predictions
        for model_name, model_preds in preds.items():
            correct_top1 = 0
            correct_topk = 0
            total = 0
            
            y_true = []
            y_pred = []
            confidences = []
            
            for arxiv_id, gt_chapter in gt_mapping.items():
                if arxiv_id in model_preds:
                    total += 1
                    pred_chapters = [c["chapter"] for c in model_preds[arxiv_id]]
                    
                    y_true.append(gt_chapter)
                    y_pred.append(pred_chapters[0] if pred_chapters else "")
                    confidences.append(model_preds[arxiv_id][0]["confidence"] if pred_chapters else 0.0)
                    
                    if pred_chapters and pred_chapters[0] == gt_chapter:
                        correct_top1 += 1
                    if gt_chapter in pred_chapters[:top_k]:
                        correct_topk += 1
            
            if total > 0:
                lower, upper = bootstrap_ci(y_true, y_pred, n_iterations=10000)
                ece, brier, bin_stats = compute_calibration(y_true, y_pred, confidences)
                
                # Plot reliability diagram
                safe_name = model_name.replace("/", "_").replace("\\", "_")
                plot_path = s_dir / "eval_plots" / f"calibration_{safe_name}.png"
                plot_reliability_diagram(bin_stats, model_name, plot_path)
                
                metrics[model_name] = {
                    "top1_acc": correct_top1 / total,
                    "top1_ci": (lower, upper),
                    "topk_recall": correct_topk / total,
                    "ece": ece,
                    "brier": brier,
                    "total": total
                }
                predictions_for_tests[model_name] = {"y_true": y_true, "y_pred": y_pred}

        # 2. Print Ablation & Calibration Table
        print(f"{'Model / Aggregation Method':<30} | {'Top-1 Acc [95% CI]':<25} | {f'Top-{top_k} Recall':<12} | {'ECE':<8} | {'Brier':<8} | {'N'}")
        print("-" * 105)
        
        for model_name in sorted(single_models):
            if model_name in metrics:
                m = metrics[model_name]
                ci_str = f"[{m['top1_ci'][0]:.1%} - {m['top1_ci'][1]:.1%}]"
                print(f"{model_name:<30} | {m['top1_acc']:>6.1%} {ci_str:<18} | {m['topk_recall']:>11.1%} | {m['ece']:>8.3f} | {m['brier']:>8.3f} | {m['total']}")
        
        print("-" * 105)
        
        aggregations = [m for m in ["Unweighted Majority", "Plurality+Tiebreaker", "_consensus"] if m in metrics]
        for model_name in aggregations:
            m = metrics[model_name]
            name_display = "Confidence-Weighted Consensus" if model_name == "_consensus" else model_name
            ci_str = f"[{m['top1_ci'][0]:.1%} - {m['top1_ci'][1]:.1%}]"
            print(f"{name_display:<30} | {m['top1_acc']:>6.1%} {ci_str:<18} | {m['topk_recall']:>11.1%} | {m['ece']:>8.3f} | {m['brier']:>8.3f} | {m['total']}")

        print(f"\nNote: Reliability plots saved to {s_dir / 'eval_plots'}")

        # 3. Paired Statistical Test
        if "_consensus" in predictions_for_tests and single_models:
            # Find best single model
            best_single_model = max(
                [m for m in single_models if m in metrics],
                key=lambda m: metrics[m]["top1_acc"],
                default=None
            )

            if best_single_model:
                cons_data = predictions_for_tests["_consensus"]
                best_data = predictions_for_tests[best_single_model]

                # Align on common papers (models may have different coverage)
                if len(cons_data["y_true"]) != len(best_data["y_true"]):
                    # Rebuild aligned vectors from GT keys both methods cover
                    cons_preds_map = dict(zip(cons_data["y_true"], cons_data["y_pred"]))
                    best_preds_map = dict(zip(best_data["y_true"], best_data["y_pred"]))
                    # Use paper identifiers stored alongside — fallback to zip by gt keys
                    common_keys = [k for k in cons_data["y_true"] if k in set(best_data["y_true"])]
                    if not common_keys:
                        common_keys = cons_data["y_true"]  # fallback
                    cons_data = {"y_true": common_keys, "y_pred": [cons_preds_map[k] for k in common_keys]}
                    best_data = {"y_true": common_keys, "y_pred": [best_preds_map.get(k, "") for k in common_keys]}

                p_val = paired_bootstrap_test(cons_data["y_true"], cons_data["y_pred"], best_data["y_pred"], n_iterations=10000)
                
                print(f"\n--- Statistical Significance (Paired Bootstrap, B=10,000) ---")
                print(f"H0: Confidence-Weighted Consensus accuracy is <= Best Single Model ({best_single_model})")
                print(f"P-value: {p_val:.4f}")
                if p_val < 0.05:
                    print("Conclusion: The consensus method is statistically significantly better (p < 0.05).")
                else:
                    print("Conclusion: Not statistically significant at the 5% level.")

if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config.yaml")
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, cli_cfg)
    main(cfg)
