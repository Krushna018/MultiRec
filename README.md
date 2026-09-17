
# MultiRec: Multi-Objective Recommender Framework


## Architecture

```text
MultiRec/
├── src/multirec/
│   ├── data.py          # 120k interaction generator + temporal split
│   ├── models.py        # ItemKNN, matrix factorization, hybrid model
│   ├── rerank.py        # relevance, weighted fusion, Pareto reranking
│   ├── metrics.py       # ranking + beyond-accuracy metrics, bootstrap CIs
│   └── experiment.py    # 12-config experiment orchestration + Pareto front
├── scripts/
│   ├── run_study.py     # full 120k-interaction study
│   ├── run_quick_demo.py
│   └── download_movielens_1m.py
├── tests/
├── docs/research_design.md
└── results/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run a quick verification

```bash
python scripts/run_quick_demo.py
```

This evaluates all **12 configurations** on a smaller deterministic dataset.

## Run the full study

```bash
python scripts/run_study.py
```

The full configuration uses:
- 1,000 users
- 1,500 items
- **120,000 interactions**
- 3 recommender models
- 4 ranking policies
- top-10 evaluation
- bootstrap confidence intervals

Outputs are written to `results/`:
- `configuration_results.csv`
- `pareto_front.csv`
- `algorithm_rankings.csv`
- `pareto_tradeoff.png`

## Why this is not a generic MovieLens recommender

The core research question is not simply whether a model can predict user
preferences. It studies **multi-objective recommendation under competing goals**:
relevance, novelty, diversity, and catalog exposure. Weighted fusion and Pareto
analysis make trade-offs explicit and measurable.

The optional MovieLens downloader exists only as a real-data extension; the project
remains fully reproducible offline without it.
