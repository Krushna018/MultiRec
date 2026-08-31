
from pathlib import Path
import sys, time
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multirec.data import generate_synthetic_interactions, temporal_train_test_split
from multirec.models import ItemKNNRecommender, MatrixFactorizationRecommender, HybridRecommender
from multirec.experiment import POLICIES, evaluate_configuration, pareto_front

def main():
    print("Generating reproducible 120,000-interaction dataset...")
    interactions, items = generate_synthetic_interactions(
        n_users=1000, n_items=1500, n_interactions=120_000, seed=42
    )
    train, test = temporal_train_test_split(interactions)
    print(f"Interactions: {len(interactions):,}; train={len(train):,}; test={len(test):,}")

    models = [
        ("ItemKNN", ItemKNNRecommender(k_neighbors=60)),
        ("MatrixFactorization", MatrixFactorizationRecommender(n_components=32, random_state=42)),
        ("Hybrid", HybridRecommender(alpha=0.70, n_components=32, random_state=42)),
    ]

    results = []
    for name, model in models:
        print(f"Fitting {name}...")
        model.fit(train, items)
        for policy in POLICIES:
            t0 = time.time()
            row, _ = evaluate_configuration(
                model, name, policy, train, test, items, k=10, n_boot=300
            )
            row["runtime_seconds"] = time.time() - t0
            results.append(row)
            print(name, policy, f"NDCG@10={row['ndcg']:.4f}")

    result_df = pd.DataFrame(results)
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    result_df.to_csv(outdir / "configuration_results.csv", index=False)

    front = pareto_front(result_df)
    front.to_csv(outdir / "pareto_front.csv", index=False)

    # Research-style ranking table.
    normalized = result_df.copy()
    metrics = ["ndcg","catalog_coverage","novelty","ild"]
    for m in metrics:
        mn, mx = normalized[m].min(), normalized[m].max()
        normalized[m+"_norm"] = (normalized[m]-mn)/(mx-mn+1e-9)
    normalized["balanced_score"] = (
        0.55*normalized["ndcg_norm"] +
        0.15*normalized["catalog_coverage_norm"] +
        0.15*normalized["novelty_norm"] +
        0.15*normalized["ild_norm"]
    )
    normalized.sort_values("balanced_score", ascending=False).to_csv(
        outdir / "algorithm_rankings.csv", index=False
    )

    # Plot relevance vs coverage and annotate configurations.
    fig, ax = plt.subplots(figsize=(8,6))
    for _, r in result_df.iterrows():
        ax.scatter(r["catalog_coverage"], r["ndcg"])
        ax.annotate(f"{r['model']}\n{r['policy']}", (r["catalog_coverage"], r["ndcg"]), fontsize=7)
    ax.set_xlabel("Catalog coverage")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Relevance–Exposure Trade-off Across 12 Configurations")
    fig.tight_layout()
    fig.savefig(outdir / "pareto_tradeoff.png", dpi=180)
    plt.close(fig)

    print("\nSaved:")
    print(outdir / "configuration_results.csv")
    print(outdir / "pareto_front.csv")
    print(outdir / "algorithm_rankings.csv")
    print(outdir / "pareto_tradeoff.png")
    print("\n12 configurations =", len(models), "models ×", len(POLICIES), "ranking policies")

if __name__ == "__main__":
    main()
