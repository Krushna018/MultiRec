
# MultiRec: Multi-Objective Recommender Framework

A reproducible research-oriented recommendation framework for studying competing
user and system objectives rather than optimizing ranking accuracy alone.

## Resume-aligned capabilities

The repository directly implements the technical scope behind these resume bullets:

- Multi-objective recommendation over **100,000+ interactions**:
  the default study generates **120,000 deterministic user-item interactions**.
- Collaborative filtering, matrix factorization, and hybrid recommendation models.
- Weighted-fusion and Pareto-inspired re-ranking.
- Exactly **12 model/ranking configurations** = 3 recommenders × 4 ranking policies.
- Precision@10, Recall@10, NDCG@10, catalog coverage, novelty, and intra-list diversity.
- User-level bootstrap confidence intervals.
- Pareto-front analysis and configuration ranking.

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

## Important resume note

The repository supports the claims about the framework, dataset scale, 12
configurations, methods, and metrics. Any sentence claiming a particular empirical
improvement should be based on the actual CSV outputs produced after running the
full study.
