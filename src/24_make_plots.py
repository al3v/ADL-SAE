from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layers_6_9_12_top100_features.csv"
FREQUENCY_PATH = PROJECT_ROOT / "outputs" / "sae_layers_6_9_12_top100_feature_frequency.csv"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def save_nonzero_feature_count_plot(sample_level: pd.DataFrame) -> None:
    summary = (
        sample_level
        .groupby(["layer", "is_correct"])["nonzero_feature_count"]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(index="layer", columns="is_correct", values="nonzero_feature_count")

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Average Nonzero SAE Features by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average nonzero SAE features")
    ax.legend(["Incorrect", "Correct"], title="Answer correctness")
    plt.tight_layout()

    output_path = PLOTS_DIR / "nonzero_sae_features_by_layer.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def save_feature_norm_plot(sample_level: pd.DataFrame) -> None:
    summary = (
        sample_level
        .groupby(["layer", "is_correct"])["feature_norm"]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(index="layer", columns="is_correct", values="feature_norm")

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Average SAE Feature Norm by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average SAE feature norm")
    ax.legend(["Incorrect", "Correct"], title="Answer correctness")
    plt.tight_layout()

    output_path = PLOTS_DIR / "sae_feature_norm_by_layer.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def save_wrong_enriched_layer12_plot(freq_df: pd.DataFrame) -> None:
    layer12 = freq_df[
        (freq_df["layer"] == 12)
        & (freq_df["is_common_feature"] == False)
    ].copy()

    top_wrong = layer12.sort_values(
        "freq_difference_wrong_minus_correct",
        ascending=False,
    ).head(15)

    labels = top_wrong["sae_feature_id"].astype(str)
    values = top_wrong["freq_difference_wrong_minus_correct"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Layer 12 Top Wrong-Enriched SAE Features")
    plt.xlabel("SAE feature ID")
    plt.ylabel("Frequency difference: incorrect - correct")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = PLOTS_DIR / "layer12_wrong_enriched_features.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def save_correct_enriched_layer12_plot(freq_df: pd.DataFrame) -> None:
    layer12 = freq_df[
        (freq_df["layer"] == 12)
        & (freq_df["is_common_feature"] == False)
    ].copy()

    top_correct = layer12.sort_values(
        "freq_difference_wrong_minus_correct",
        ascending=True,
    ).head(15)

    labels = top_correct["sae_feature_id"].astype(str)
    values = -top_correct["freq_difference_wrong_minus_correct"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Layer 12 Top Correct-Enriched SAE Features")
    plt.xlabel("SAE feature ID")
    plt.ylabel("Frequency difference: correct - incorrect")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = PLOTS_DIR / "layer12_correct_enriched_features.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(FEATURES_PATH)
    freq_df = pd.read_csv(FREQUENCY_PATH)

    sample_level = features_df.drop_duplicates(["sample_id", "layer"]).copy()

    print("Samples:", sample_level["sample_id"].nunique())
    print("Layers:", sorted(sample_level["layer"].unique()))
    print()

    print("Average nonzero SAE features:")
    print(sample_level.groupby(["layer", "is_correct"])["nonzero_feature_count"].mean())
    print()

    print("Average SAE feature norm:")
    print(sample_level.groupby(["layer", "is_correct"])["feature_norm"].mean())
    print()

    save_nonzero_feature_count_plot(sample_level)
    save_feature_norm_plot(sample_level)
    save_wrong_enriched_layer12_plot(freq_df)
    save_correct_enriched_layer12_plot(freq_df)


if __name__ == "__main__":
    main()
