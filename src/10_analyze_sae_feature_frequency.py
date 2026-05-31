from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layer1_top_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sae_layer1_feature_frequency.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Count how often each SAE feature appears in top-k lists
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

    # Ensure both columns exist
    if True not in pivot.columns:
        pivot[True] = 0
    if False not in pivot.columns:
        pivot[False] = 0

    pivot = pivot.rename(
        columns={
            True: "count_correct",
            False: "count_wrong",
        }
    )

    num_correct_samples = df[df["is_correct"] == True]["sample_id"].nunique()
    num_wrong_samples = df[df["is_correct"] == False]["sample_id"].nunique()

    pivot["freq_correct"] = pivot["count_correct"] / num_correct_samples
    pivot["freq_wrong"] = pivot["count_wrong"] / num_wrong_samples

    pivot["freq_difference_wrong_minus_correct"] = (
        pivot["freq_wrong"] - pivot["freq_correct"]
    )

    pivot = pivot.sort_values(
        "freq_difference_wrong_minus_correct",
        ascending=False,
    )

    pivot.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved SAE feature frequency analysis to: {OUTPUT_PATH}")
    print()
    print("Features most enriched in wrong cases:")
    print(
        pivot[
            [
                "sae_feature_id",
                "count_wrong",
                "count_correct",
                "freq_wrong",
                "freq_correct",
                "freq_difference_wrong_minus_correct",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Features most enriched in correct cases:")
    print(
        pivot[
            [
                "sae_feature_id",
                "count_wrong",
                "count_correct",
                "freq_wrong",
                "freq_correct",
                "freq_difference_wrong_minus_correct",
            ]
        ]
        .tail(20)
        .sort_values("freq_difference_wrong_minus_correct")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
