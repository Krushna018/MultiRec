
from __future__ import annotations
import numpy as np

def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if finite.any():
        vals = x[finite]
        out[finite] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    return out

def rerank_relevance(scores: np.ndarray, k: int = 10) -> list[int]:
    idx = np.argsort(scores)[::-1]
    return [int(i) for i in idx if np.isfinite(scores[i])][:k]

def rerank_weighted(
    scores: np.ndarray,
    popularity: np.ndarray,
    item_features: np.ndarray,
    k: int = 10,
    objective: str = "diversity",
    relevance_weight: float = 0.75,
) -> list[int]:
    """Greedy weighted reranking for relevance + diversity or novelty."""
    relevance = _minmax(scores)
    pop = popularity / (popularity.max() + 1e-9)
    novelty = 1.0 - pop
    candidates = [int(i) for i in np.argsort(scores)[::-1] if np.isfinite(scores[i])][:max(100, k*10)]
    selected = []
    X = item_features.astype(float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    while candidates and len(selected) < k:
        best, best_val = None, -1e18
        for i in candidates:
            if objective == "novelty":
                secondary = novelty[i]
            else:
                if not selected:
                    secondary = 1.0
                else:
                    sims = Xn[selected] @ Xn[i]
                    secondary = 1.0 - float(np.mean(sims))
            val = relevance_weight * relevance[i] + (1-relevance_weight) * secondary
            if val > best_val:
                best, best_val = i, val
        selected.append(best)
        candidates.remove(best)
    return selected

def rerank_pareto(
    scores: np.ndarray,
    popularity: np.ndarray,
    item_features: np.ndarray,
    k: int = 10,
) -> list[int]:
    """Pareto-inspired greedy reranking across relevance, novelty, and diversity."""
    relevance = _minmax(scores)
    novelty = 1.0 - popularity / (popularity.max() + 1e-9)
    candidates = [int(i) for i in np.argsort(scores)[::-1] if np.isfinite(scores[i])][:max(120, k*12)]
    X = item_features.astype(float)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    selected = []

    while candidates and len(selected) < k:
        pts = []
        for i in candidates:
            if selected:
                div = 1.0 - float(np.mean(Xn[selected] @ Xn[i]))
            else:
                div = 1.0
            pts.append((i, relevance[i], novelty[i], div))
        # Non-dominated subset.
        front = []
        for a in pts:
            dominated = False
            for b in pts:
                if b[0] == a[0]:
                    continue
                if (b[1] >= a[1] and b[2] >= a[2] and b[3] >= a[3] and
                    (b[1] > a[1] or b[2] > a[2] or b[3] > a[3])):
                    dominated = True
                    break
            if not dominated:
                front.append(a)
        # Choose balanced point from the front.
        best = max(front, key=lambda t: 0.55*t[1] + 0.225*t[2] + 0.225*t[3])
        selected.append(best[0])
        candidates.remove(best[0])
    return selected
