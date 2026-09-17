# MultiRec — Multi-Objective Recommender Framework

<p align="center">
  <b>A reproducible recommendation framework for studying trade-offs between relevance, novelty, diversity, and catalog exposure.</b>
</p>


---

## Overview

**MultiRec** is a reproducible Python framework for evaluating recommender systems under **multiple competing objectives**.

Instead of optimizing recommendation quality using relevance alone, MultiRec studies the interaction between:

* **Relevance** — how well recommendations match user preferences
* **Novelty** — how much recommendations differ from familiar/popular items
* **Diversity** — how varied the recommended items are
* **Catalog Exposure** — how broadly the item catalog is represented

The framework compares multiple recommendation models and reranking strategies under a common experimental protocol, making objective trade-offs explicit through **weighted fusion and Pareto analysis**.

---

## Research Setup

The default study evaluates:

| Component            |                      Configuration |
| -------------------- | ---------------------------------: |
| Users                |                          **1,000** |
| Items                |                          **1,500** |
| Interactions         |                        **120,000** |
| Recommender models   |                              **3** |
| Ranking policies     |                              **4** |
| Total configurations |                             **12** |
| Evaluation           |                         **Top-10** |
| Uncertainty          | **Bootstrap confidence intervals** |

### Experimental pipeline

```text
User–Item Interactions
          │
          ▼
     Temporal Split
          │
          ▼
   ┌───────────────┐
   │ Recommendation│
   │    Models     │
   └───────┬───────┘
           │
           ▼
      Reranking
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
Relevance Novelty Diversity
    │      │      │
    └──────┼──────┘
           ▼
    Catalog Exposure
           │
           ▼
   Multi-Objective Results
           │
           ▼
     Pareto Analysis
```

---

## Models & Ranking Strategies

### Recommendation Models

* **ItemKNN**
* **Matrix Factorization**
* **Hybrid recommender**

### Ranking / Reranking Policies

The framework supports multiple ranking strategies based on relevance and beyond-accuracy objectives, including:

* relevance-based ranking
* weighted objective fusion
* Pareto-based reranking

The combination produces **12 experimental configurations** for systematic comparison.

---

## Architecture

```text
MultiRec/
├── src/multirec/
│   ├── data.py          # Interaction generation + temporal split
│   ├── models.py        # ItemKNN, matrix factorization, hybrid model
│   ├── rerank.py        # Relevance, weighted fusion, Pareto reranking
│   ├── metrics.py       # Ranking metrics + bootstrap confidence intervals
│   └── experiment.py    # Experiment orchestration + Pareto analysis
│
├── scripts/
│   ├── run_study.py
│   ├── run_quick_demo.py
│   └── download_movielens_1m.py
│
├── tests/
├── docs/
│   └── research_design.md
├── results/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## Quick Demo

Run a lightweight deterministic experiment:

```bash
python scripts/run_quick_demo.py
```

This evaluates all **12 configurations** on a smaller dataset and verifies the complete recommendation and evaluation pipeline.

---

## Full Study

Run the complete experiment:

```bash
python scripts/run_study.py
```

Results are saved to `results/`:

| Output                      | Description                                 |
| --------------------------- | ------------------------------------------- |
| `configuration_results.csv` | Metrics for all experimental configurations |
| `pareto_front.csv`          | Non-dominated configurations                |
| `algorithm_rankings.csv`    | Comparative configuration rankings          |
| `pareto_tradeoff.png`       | Visualization of objective trade-offs       |

---

## Reproducibility

MultiRec is designed to run **fully offline** using its deterministic synthetic interaction generator.

The optional MovieLens downloader provides a real-world dataset extension:

```bash
python scripts/download_movielens_1m.py
```

The MovieLens dataset is therefore **not required** to reproduce the core study.

---

## Research Focus

MultiRec does not treat recommendation as a single-metric prediction problem.

Its central question is:

> **How do recommendation models and reranking strategies behave when relevance must be balanced against novelty, diversity, and catalog exposure?**

Rather than collapsing these objectives into one score, the framework uses **multi-objective evaluation and Pareto analysis** to make the trade-offs visible.
