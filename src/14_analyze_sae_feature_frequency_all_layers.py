from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layers_1_2_3_top_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sae_layers_1_2_3_feature_frequency.csv"

COMMON_FEATURE_THRESHOLD = 0.80


def analyze_layer(df: pd.DataFrame, layer: int) -> pd.DataFrame:
    layer_df = df[df["layer"] == layer].copy()

    num_correct_samples = layer_df[layer_df["is_correct"] == True]["sample_id"].nunique()
    num_wrong_samples = layer_df[layer_df["is_correct"] == False]["sample_id"].nunique()
    num_total_samples = layer_df["sample_id"].nunique()

    grouped = (
        layer_df.groupby(["sae_feature_id", "is_correct"])
        .size()
        .reset_index(name="count")
    )

    pivot = grouped.pivot_table(
        index="sae_feature_id",
        columns="is_correct",
        values="count",
        fill_value=0,
    ).reset_index()

    if True not in pivot.columns:
        pivot[True] = 0
    if False not in pivot.columns:
        pivot[False] = 0

    pivot = pivot.rename(columns={True: "count_correct", False: "count_wrong"})

    pivot["layer"] = layer
    pivot["num_correct_samples"] = num_correct_samples
    pivot["num_wrong_samples"] = num_wrong_samples
    pivot["num_total_samples"] = num_total_samples

    pivot["freq_correct"] = pivot["count_correct"] / num_correct_samples
    pivot["freq_wrong"] = pivot["count_wrong"] / num_wrong_samples
    pivot["total_count"] = pivot["count_correct"] + pivot["count_wrong"]
    pivot["freq_total"] = pivot["total_count"] / num_total_samples

    pivot["freq_difference_wrong_minus_correct"] = (
        pivot["freq_wrong"] - pivot["freq_correct"]
    )

    pivot["is_common_feature"] = pivot["freq_total"] >= COMMON_FEATURE_THRESHOLD

    return pivot


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    all_layers = []

    for layer in sorted(df["layer"].unique()):
        all_layers.append(analyze_layer(df, layer))

    out_df = pd.concat(all_layers, ignore_index=True)
    out_df = out_df.sort_values(
        ["layer", "freq_difference_wrong_minus_correct"],
        ascending=[True, False],
    )

    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved all-layer feature frequency analysis to: {OUTPUT_PATH}")
    print()

    for layer in sorted(out_df["layer"].unique()):
        layer_df = out_df[out_df["layer"] == layer]
        filtered = layer_df[layer_df["is_common_feature"] == False]

        print("=" * 80)
        print(f"Layer {layer}")
        print("=" * 80)
        print(f"Total features in top-k lists: {len(layer_df)}")
        print(f"Non-common features: {len(filtered)}")
        print()

        print("Most wrong-enriched non-common features:")
        print(
            filtered[
                [
                    "sae_feature_id",
                    "count_wrong",
                    "count_correct",
                    "freq_wrong",
                    "freq_correct",
                    "freq_total",
                    "freq_difference_wrong_minus_correct",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        print()


if __name__ == "__main__":
    main()
