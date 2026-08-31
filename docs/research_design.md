
# Research Design

## Research question
How do multi-objective re-ranking strategies alter the trade-off between user
relevance and system-level objectives such as diversity, novelty, and long-tail
catalog exposure?

## Experimental factors
Three base recommendation models:
1. Item-based collaborative filtering
2. Low-rank matrix-factorization model
3. Hybrid collaborative + content model

Four ranking policies:
1. Relevance only
2. Relevance + diversity weighted fusion
3. Relevance + novelty weighted fusion
4. Pareto-inspired relevance/novelty/diversity re-ranking

This yields exactly **12 model/ranking configurations**.

## Dataset
The default offline experiment uses a deterministic synthetic interaction generator
with:
- 1,000 users
- 1,500 items
- **120,000 interactions**
- long-tail popularity
- latent user preference structure
- multi-label genre/content features

An optional downloader is included for MovieLens 1M (1,000,209 ratings).

## Evaluation
Ranking effectiveness:
- Precision@10
- Recall@10
- NDCG@10

Beyond-accuracy objectives:
- Catalog coverage
- Novelty (self-information based on item popularity)
- Intra-list diversity based on content similarity

Uncertainty:
- User-level bootstrap confidence intervals

Multi-objective analysis:
- Pareto-front extraction across NDCG@10, coverage, novelty, and diversity
- Balanced configuration ranking for exploratory comparison

## Leakage control
Interactions are split chronologically within each user so evaluation items occur
after training interactions in the generated timeline.

## Reproducibility
All generators/models accept fixed seeds, configuration names are explicit, and
full output tables are written to CSV.
