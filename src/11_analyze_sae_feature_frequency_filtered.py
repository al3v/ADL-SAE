from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layer1_top_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sae_layer1_feature_frequency_filtered.csv"

COMMON_FEATURE_THRESHOLD = 0.80


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    num_correct_samples = df[df["is_correct"] == True]["sample_id"].nunique()
    num_wrong_samples = df[df["is_correct"] == False]["sample_id"].nunique()
    num_total_samples = df["sample_id"].nunique()

    grouped = (
        df.groupby(["sae_feature_id", "is_correct"])
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

    pivot["freq_correct"] = pivot["count_correct"] / num_correct_samples
    pivot["freq_wrong"] = pivot["count_wrong"] / num_wrong_samples
    pivot["total_count"] = pivot["count_correct"] + pivot["count_wrong"]
    pivot["freq_total"] = pivot["total_count"] / num_total_samples

    pivot["freq_difference_wrong_minus_correct"] = (
        pivot["freq_wrong"] - pivot["freq_correct"]
    )

    filtered = pivot[pivot["freq_total"] < COMMON_FEATURE_THRESHOLD].copy()
    filtered = filtered.sort_values(
        "freq_difference_wrong_minus_correct",
        ascending=False,
    )

    filtered.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved filtered SAE feature analysis to: {OUTPUT_PATH}")
    print()

    print(f"Total features before filtering: {len(pivot)}")
    print(f"Total features after filtering: {len(filtered)}")
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
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Most correct-enriched non-common features:")
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
        .tail(20)
        .sort_values("freq_difference_wrong_minus_correct")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
