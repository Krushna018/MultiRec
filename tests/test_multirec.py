
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from multirec.data import generate_synthetic_interactions, temporal_train_test_split
from multirec.models import ItemKNNRecommender, MatrixFactorizationRecommender, HybridRecommender
from multirec.rerank import rerank_relevance, rerank_weighted, rerank_pareto
from multirec.metrics import precision_at_k, recall_at_k, ndcg_at_k
from multirec.experiment import POLICIES

def fixture():
    interactions, items = generate_synthetic_interactions(
        n_users=20, n_items=60, n_interactions=600, seed=3
    )
    return interactions, items, *temporal_train_test_split(interactions)

def test_default_dataset_claim():
    interactions, _ = generate_synthetic_interactions(
        n_users=1000, n_items=1500, n_interactions=120_000, seed=42
    )
    assert len(interactions) == 120_000
    assert len(interactions) > 100_000

def test_twelve_configurations():
    assert 3 * len(POLICIES) == 12

def test_models_fit_and_score():
    _, items, train, _ = fixture()
    for m in [ItemKNNRecommender(10),
              MatrixFactorizationRecommender(8, 3),
              HybridRecommender(0.7, 8, 3)]:
        m.fit(train, items)
        scores = m.score_user(0)
        assert len(scores) == len(items)

def test_rerankers_return_unique_items():
    scores = np.arange(30, dtype=float)
    pop = np.arange(1,31,dtype=float)
    X = np.eye(30, 5)
    for rec in [
        rerank_relevance(scores, 10),
        rerank_weighted(scores, pop, X, 10, "novelty"),
        rerank_weighted(scores, pop, X, 10, "diversity"),
        rerank_pareto(scores, pop, X, 10),
    ]:
        assert len(rec) == 10
        assert len(set(rec)) == 10

def test_metrics():
    rec = [1,2,3,4,5]
    rel = {2,3,9}
    assert 0 <= precision_at_k(rec, rel, 5) <= 1
    assert 0 <= recall_at_k(rec, rel, 5) <= 1
    assert 0 <= ndcg_at_k(rec, rel, 5) <= 1
