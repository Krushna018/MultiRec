
from __future__ import annotations
import numpy as np
import pandas as pd
from .rerank import rerank_relevance, rerank_weighted, rerank_pareto
from .metrics import evaluate_topk, bootstrap_ci

POLICIES = ["relevance", "weighted_diversity", "weighted_novelty", "pareto"]

def build_recommendations(model, train, item_df, policy, k=10):
    n_items = item_df.shape[0]
    counts = train.groupby("item_id").size()
    popularity = np.zeros(n_items)
    for i, c in counts.items():
        popularity[int(i)] = c
    feat_cols = [c for c in item_df.columns if c.startswith("genre_")]
    X = item_df.sort_values("item_id")[feat_cols].to_numpy(float)

    recs = {}
    for u in sorted(train.user_id.unique()):
        scores = model.score_user(int(u))
        if policy == "relevance":
            rec = rerank_relevance(scores, k)
        elif policy == "weighted_diversity":
            rec = rerank_weighted(scores, popularity, X, k, objective="diversity", relevance_weight=0.75)
        elif policy == "weighted_novelty":
            rec = rerank_weighted(scores, popularity, X, k, objective="novelty", relevance_weight=0.75)
        elif policy == "pareto":
            rec = rerank_pareto(scores, popularity, X, k)
        else:
            raise ValueError(policy)
        recs[int(u)] = rec
    return recs

def evaluate_configuration(model, model_name, policy, train, test, item_df, k=10, n_boot=300):
    recs = build_recommendations(model, train, item_df, policy, k)
    feat_cols = [c for c in item_df.columns if c.startswith("genre_")]
    X = item_df.sort_values("item_id")[feat_cols].to_numpy(float)
    summary, per_user = evaluate_topk(recs, test, X, train, k)
    row = {"model": model_name, "policy": policy, **summary}
    for metric in ["precision","recall","ndcg","novelty","ild"]:
        lo, hi = bootstrap_ci(per_user[metric].to_numpy(), n_boot=n_boot, seed=42)
        row[f"{metric}_ci_low"] = lo
        row[f"{metric}_ci_high"] = hi
    return row, per_user

def pareto_front(df: pd.DataFrame, metrics=("ndcg","catalog_coverage","novelty","ild")) -> pd.DataFrame:
    vals = df[list(metrics)].to_numpy(float)
    keep = []
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    out = df.iloc[keep].copy()
    out["pareto_optimal"] = True
    return out.sort_values("ndcg", ascending=False)
