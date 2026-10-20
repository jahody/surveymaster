# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SurveyMaster** — a Python pipeline that classifies research papers against hierarchical taxonomies using multi-model LLM consensus voting. Different LLMs have complementary organizational intuitions; aggregating their confidence-weighted votes produces more reliable classifications than any single model alone. Human-in-the-loop gates at taxonomy design and evaluation ensure quality. An optional refinement loop in `4_evaluate.py` measures taxonomy coverage and re-prompts the refiner LLM to fix dead or overcrowded leaf categories.

Bundled data: three topics under `papers/` (`efficient-reasoning-LLM`, `implicit-reasoning-LLM`, `latent-reasoning`) with abstracts, the taxonomy, survey ground truth, per-model classifications, consensus and evaluation.

## Setup

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```
OPENROUTER_API_KEY=...
ASTA_API_KEY=...
```

## Running

```bash
python app.py            # Web UI at http://localhost:8765 — start first, keep open
```

Pipeline stages (run in order):
```bash
python 1_fetch.py        # Fetch papers (+ blog scraping if source=blogPosts)
python 2_taxonomize.py   # Propose axes → 👤 select → generate taxonomies → 👤 review
python 3_classify.py     # Classify papers + aggregate consensus
python 4_evaluate.py     # Evaluate metrics → 👤 accept or refine
python 5_evolve.py       # Generate paper influence/evolution tree
```

### Web app port

The whole pipeline also runs in the browser at `jahody.github.io/surveymaster/`. The source is [`src/surveymaster/`](https://github.com/jahody/jahody.github.io/tree/main/src/surveymaster) in the website repo, one TypeScript module per stage with the prompts copied verbatim. Its consensus voting and all 8 metrics match this code's output exactly on the exported topics. When you change prompts, voting or metrics here, port the change there.

```bash
python export_web.py     # Export finished topics as the web app's demo projects (stdlib only; writes to ../jahody.github.io by default, or --out DIR)
```

## Configuration

All pipeline settings live in `config.yaml` (Hydra/OmegaConf): models to use, API endpoints, concurrency limits, top-k parameters, topic directory, paper content source (`abstracts` | `blogPosts` | `papers_markdown`).

Key config sections: `generation`, `refinement`, `taxonomy`, `evolution`, `evaluation`.

## Architecture

### Pipeline

```
Fetch → Taxonomize (👤 select axes → 👤 review) → Classify + Aggregate → Evaluate (👤 accept or refine) → Evolution Tree
```

**Stage 1** — `1_fetch.py`: Fetches arXiv papers via Asta MCP or Semantic Scholar. Three content sources: `abstracts` (metadata only), `blogPosts` (AlphaXiv summaries via Playwright), `papers_markdown` (full text). Blog scraping runs automatically when `source: blogPosts`. Stored under `papers/{topic}/`.

**Stage 2** — `2_taxonomize.py`: LLMs propose taxonomy axes → 👤 human selects + adds comments → LLM generates full taxonomies → 👤 human reviews in `/editor`. Saves approved axes to `axes_approved.json` and taxonomies to `papers/{topic}/taxonomies/*.json`.

**Stage 3** — `3_classify.py`: Each paper × each taxonomy × each model run in parallel (AsyncOpenAI + semaphore). Each model returns top-k chapters with integer confidence % summing to 100. Then aggregates via confidence-weighted majority voting. Output: `papers/{topic}/classification/{model}/*.json` + `_consensus/*.json`.

**Stage 4** — `4_evaluate.py`: Computes 8 metrics and generates plots (see table below). 👤 Human decides to accept or auto-refine. If refining, the script fixes dead/fat leaves via LLM re-prompting, then user reviews and reclassifies.

**Stage 5** — `5_evolve.py`: Same two-phase approach for generating paper influence/lineage trees using chronological arXiv metadata.

### Evaluation Metrics (`4_evaluate.py`)

| # | Metric | Function |
|---|--------|---------|
| 1 | Pairwise inter-model agreement + Cohen's kappa | `compute_pairwise_agreement` |
| 2 | Fleiss' kappa (multi-rater) | `compute_fleiss_kappa` |
| 3 | Consensus strength (winning margin) | `compute_consensus_strength` |
| 4 | Confidence–agreement Pearson r | `compute_confidence_agreement` |
| 5 | Per-model deviation from consensus | `compute_model_deviation` |
| 6 | Per-taxonomy difficulty (unanimous/majority rates, entropy) | `compute_taxonomy_difficulty` |
| 7 | Coverage analysis (papers per leaf, dead/fat counts, Gini, entropy) | `compute_coverage_analysis` |
| 8 | Cross-taxonomy independence (Cramér's V) | `compute_taxonomy_independence` |

Output: `papers/{topic}/evaluation/consistency_metrics.json` + 6 PNG plots.

### Web UI (Flask)

`app.py` serves on port 8765 with Jinja2 templates in `templates/`:

| Route | Purpose |
|-------|---------|
| `/` | Home |
| `/papers` | Browse papers by topic |
| `/editor` | Curate taxonomy JSON |
| `/classification` | Compare model predictions |
| `/evaluation` | Consistency metrics and plots |
| `/evolution-tree` | Paper influence visualization |

API routes follow `/api/...` — serve JSON from the `papers/` directory.

### Data Layout

```
papers/{topic}/
  abstracts/           # Paper metadata JSONs
  blogPosts/           # Blog post summaries (gitignored: third-party content)
  papers_markdown/     # Full paper markdown (gitignored: third-party content)
  taxonomies/          # by_technical_paradigm.json, by_application_domain.json, ...
  taxonomies_GT/       # Ground truth taxonomies (*-GT.json)
  classification/
    {provider}/{model}/*.json   # Per-model predictions
    _consensus/*.json           # Aggregated results
  evolution/
    evolution_tree.json
  evaluation/
    consistency_metrics.json
    plots/             # agreement_heatmap.png, taxonomy_independence.png, ...
  search_results.json
prompts/               # Prompts for LLM operations (e.g., GT extraction)
```

### Key Data Shapes

**Taxonomy node** (recursive):
```json
{
  "chapter": "1.2",
  "name": "category_name",
  "description": "...",
  "phrases": ["key phrase"],
  "hard_negative": "What does NOT belong here",
  "items": [...]
}
```

**Classification output** (per paper per model):
```json
{
  "filename": "2408.08435.md",
  "classifications": {
    "taxonomy_name": {
      "chapters": [{"chapter": "1.2.1", "confidence": 75, "reason": "..."}, ...],
      "chapter": "1.2.1",
      "confidence": 75
    }
  }
}
```

**Consensus output**:
```json
{
  "filename": "2408.08435.md",
  "classifications": {
    "taxonomy_name": {
      "chapter": "1.2.1",
      "score": 245,
      "votes": {"1.2.1": 245, "2.1": 80},
      "reason": "Consensus: 3/3 models ranked this #1"
    }
  }
}
```

## Key Dependencies

- **Hydra + OmegaConf** — configuration
- **AsyncOpenAI + httpx** — async LLM calls via OpenRouter
- **Flask** — web server
- **Playwright** — browser scraping for blog posts
- **python-dotenv** — environment loading
- **markdownify** — HTML to Markdown conversion (fallback for arXiv HTML)
- **matplotlib** — evaluation plots (optional)
