
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def _matrix(train: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (train["value"].to_numpy(float), (train["user_id"], train["item_id"])),
        shape=(n_users, n_items)
    )

class ItemKNNRecommender:
    """Item-based collaborative filtering using cosine similarity."""
    name = "item_knn"

    def __init__(self, k_neighbors: int = 60):
        self.k_neighbors = k_neighbors

    def fit(self, train: pd.DataFrame, item_features: pd.DataFrame):
        self.n_users = int(train.user_id.max()) + 1
        self.n_items = int(item_features.item_id.max()) + 1
        self.R = _matrix(train, self.n_users, self.n_items)
        sim = cosine_similarity(self.R.T, dense_output=True)
        np.fill_diagonal(sim, 0.0)
        # retain top-k similarities per item
        if self.k_neighbors < self.n_items:
            keep = np.argpartition(sim, -self.k_neighbors, axis=1)[:, -self.k_neighbors:]
            mask = np.zeros_like(sim, dtype=bool)
            rows = np.arange(self.n_items)[:, None]
            mask[rows, keep] = True
            sim = np.where(mask, sim, 0.0)
        self.sim = sim
        return self

    def score_user(self, user_id: int) -> np.ndarray:
        profile = self.R[user_id].toarray().ravel()
        scores = profile @ self.sim
        seen = profile > 0
        scores[seen] = -np.inf
        return scores

class MatrixFactorizationRecommender:
    """Low-rank collaborative model using Truncated SVD."""
    name = "matrix_factorization"

    def __init__(self, n_components: int = 32, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, train: pd.DataFrame, item_features: pd.DataFrame):
        self.n_users = int(train.user_id.max()) + 1
        self.n_items = int(item_features.item_id.max()) + 1
        self.R = _matrix(train, self.n_users, self.n_items)
        k = min(self.n_components, max(2, min(self.R.shape)-1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(self.R)
        self.item_factors = self.svd.components_.T
        return self

    def score_user(self, user_id: int) -> np.ndarray:
        scores = self.user_factors[user_id] @ self.item_factors.T
        seen = self.R[user_id].toarray().ravel() > 0
        scores[seen] = -np.inf
        return scores

class HybridRecommender:
    """Hybrid model blending collaborative latent scores and content similarity."""
    name = "hybrid"

    def __init__(self, alpha: float = 0.70, n_components: int = 32, random_state: int = 42):
        self.alpha = alpha
        self.mf = MatrixFactorizationRecommender(n_components, random_state)

    def fit(self, train: pd.DataFrame, item_features: pd.DataFrame):
        self.mf.fit(train, item_features)
        self.R = self.mf.R
        feat_cols = [c for c in item_features.columns if c.startswith("genre_")]
        self.X = item_features.sort_values("item_id")[feat_cols].to_numpy(float)
        norms = np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-9
        self.Xn = self.X / norms
        return self

    def score_user(self, user_id: int) -> np.ndarray:
        mf_scores = self.mf.score_user(user_id).copy()
        seen_idx = self.R[user_id].indices
        if len(seen_idx):
            profile = self.Xn[seen_idx].mean(axis=0)
            profile /= np.linalg.norm(profile) + 1e-9
            content = self.Xn @ profile
        else:
            content = np.zeros(self.Xn.shape[0])

        finite = np.isfinite(mf_scores)
        m = mf_scores[finite]
        if len(m):
            mf_norm = np.zeros_like(mf_scores, dtype=float)
            mf_norm[finite] = (m - m.min()) / (m.max() - m.min() + 1e-9)
        else:
            mf_norm = np.zeros_like(mf_scores, dtype=float)
        scores = self.alpha * mf_norm + (1-self.alpha) * content
        scores[~finite] = -np.inf
        return scores
