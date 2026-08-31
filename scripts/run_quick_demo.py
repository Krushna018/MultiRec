
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multirec.data import generate_synthetic_interactions, temporal_train_test_split
from multirec.models import ItemKNNRecommender, MatrixFactorizationRecommender, HybridRecommender
from multirec.experiment import POLICIES, evaluate_configuration

def main():
    interactions, items = generate_synthetic_interactions(
        n_users=80, n_items=180, n_interactions=4_000, seed=7
    )
    train, test = temporal_train_test_split(interactions)
    models = [
        ("ItemKNN", ItemKNNRecommender(k_neighbors=25)),
        ("MatrixFactorization", MatrixFactorizationRecommender(n_components=12, random_state=7)),
        ("Hybrid", HybridRecommender(alpha=0.7, n_components=12, random_state=7)),
    ]
    rows = []
    for name, m in models:
        m.fit(train, items)
        for p in POLICIES:
            row, _ = evaluate_configuration(m, name, p, train, test, items, k=10, n_boot=50)
            rows.append(row)
    df = pd.DataFrame(rows)
    print(df[["model","policy","precision","recall","ndcg","catalog_coverage","novelty","ild"]].to_string(index=False))
    print(f"\nConfigurations evaluated: {len(df)}")

if __name__ == "__main__":
    main()
