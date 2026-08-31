
from __future__ import annotations
import numpy as np
import pandas as pd

def precision_at_k(rec, relevant, k=10):
    rec = rec[:k]
    return len(set(rec) & relevant) / max(k, 1)

def recall_at_k(rec, relevant, k=10):
    return len(set(rec[:k]) & relevant) / max(len(relevant), 1)

def ndcg_at_k(rec, relevant, k=10):
    gains = np.array([1.0 if i in relevant else 0.0 for i in rec[:k]])
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal_len = min(len(relevant), k)
    if ideal_len == 0:
        return 0.0
    idcg = float(discounts[:ideal_len].sum())
    return dcg / idcg

def novelty(rec, popularity_prob):
    vals = [max(popularity_prob[i], 1e-12) for i in rec]
    return float(np.mean([-np.log2(v) for v in vals])) if vals else 0.0

def intra_list_diversity(rec, item_features):
    if len(rec) < 2:
        return 0.0
    X = item_features[rec].astype(float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = X @ X.T
    tri = sim[np.triu_indices(len(rec), k=1)]
    return float(1.0 - tri.mean()) if len(tri) else 0.0

def evaluate_topk(recommendations, test, item_features, train, k=10):
    relevant = {u: set(g.item_id.astype(int)) for u, g in test.groupby("user_id")}
    pop_counts = train.groupby("item_id").size()
    n_items = item_features.shape[0]
    pop_prob = np.zeros(n_items)
    for i, c in pop_counts.items():
        pop_prob[int(i)] = c / len(train)

    rows, all_rec = [], set()
    for u, rec in recommendations.items():
        rel = relevant.get(u, set())
        all_rec.update(rec[:k])
        rows.append({
            "user_id": u,
            "precision": precision_at_k(rec, rel, k),
            "recall": recall_at_k(rec, rel, k),
            "ndcg": ndcg_at_k(rec, rel, k),
            "novelty": novelty(rec[:k], pop_prob),
            "ild": intra_list_diversity(rec[:k], item_features),
        })
    per_user = pd.DataFrame(rows)
    summary = per_user.mean(numeric_only=True).to_dict()
    summary["catalog_coverage"] = len(all_rec) / n_items
    return summary, per_user

def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(rng.choice(values, size=len(values), replace=True).mean())
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)
