
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

GENRES = ["Action","Adventure","Animation","Comedy","Crime","Documentary",
          "Drama","Fantasy","Horror","Romance","SciFi","Thriller"]

def generate_synthetic_interactions(
    n_users: int = 1000,
    n_items: int = 1500,
    n_interactions: int = 120_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a deterministic implicit-feedback dataset with content features.

    The default configuration contains 120,000 user-item interactions, satisfying
    the resume claim of 100,000+ interactions while remaining fully reproducible
    and independent of external downloads.
    """
    if n_interactions <= n_users * 5:
        raise ValueError("n_interactions must leave enough interactions per user")

    rng = np.random.default_rng(seed)
    latent_dim = 12

    user_latent = rng.normal(size=(n_users, latent_dim))
    item_latent = rng.normal(size=(n_items, latent_dim))

    # Multi-hot item genre/content matrix.
    item_genres = np.zeros((n_items, len(GENRES)), dtype=np.int8)
    for i in range(n_items):
        count = int(rng.integers(1, 4))
        idx = rng.choice(len(GENRES), size=count, replace=False)
        item_genres[i, idx] = 1

    # Popularity prior creates a realistic long tail.
    ranks = np.arange(1, n_items + 1, dtype=float)
    popularity = 1.0 / np.power(ranks, 0.82)
    popularity /= popularity.sum()

    rows = []
    per_user = n_interactions // n_users
    remainder = n_interactions % n_users

    for u in range(n_users):
        target = per_user + (1 if u < remainder else 0)
        # Candidate pool combines popularity and latent preference.
        cand_size = min(n_items, max(target * 5, 250))
        candidates = rng.choice(n_items, size=cand_size, replace=False, p=popularity)
        logits = item_latent[candidates] @ user_latent[u]
        logits = (logits - logits.mean()) / (logits.std() + 1e-8)
        probs = np.exp(np.clip(logits, -4, 4))
        probs /= probs.sum()
        chosen = rng.choice(candidates, size=min(target, len(candidates)), replace=False, p=probs)
        # If target exceeds sampled pool, fill from unseen.
        if len(chosen) < target:
            unseen = np.setdiff1d(np.arange(n_items), chosen, assume_unique=False)
            extra = rng.choice(unseen, size=target-len(chosen), replace=False)
            chosen = np.concatenate([chosen, extra])
        timestamps = rng.integers(1_600_000_000, 1_700_000_000, size=target)
        for i, ts in zip(chosen, timestamps):
            rows.append((u, int(i), 1.0, int(ts)))

    interactions = pd.DataFrame(rows, columns=["user_id","item_id","value","timestamp"])
    interactions = interactions.sort_values(["user_id","timestamp"]).reset_index(drop=True)

    item_df = pd.DataFrame({"item_id": np.arange(n_items)})
    for j, g in enumerate(GENRES):
        item_df[f"genre_{g}"] = item_genres[:, j]

    return interactions, item_df

def temporal_train_test_split(
    interactions: pd.DataFrame,
    test_fraction: float = 0.2,
    min_train: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split to reduce future-to-past leakage."""
    train_parts, test_parts = [], []
    for uid, grp in interactions.sort_values("timestamp").groupby("user_id"):
        n = len(grp)
        cut = max(min_train, int(round(n * (1 - test_fraction))))
        cut = min(cut, n - 1)
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)
