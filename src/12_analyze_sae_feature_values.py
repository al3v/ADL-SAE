from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layer1_top_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sae_layer1_feature_value_analysis.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    grouped = (
        df.groupby(["sae_feature_id", "is_correct"])
        .agg(
            count=("sample_id", "count"),
            mean_value=("sae_feature_value", "mean"),
            max_value=("sae_feature_value", "max"),
            mean_rank=("feature_rank", "mean"),
        )
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index="sae_feature_id",
        columns="is_correct",
        values=["count", "mean_value", "max_value", "mean_rank"],
        fill_value=0,
    )

    pivot.columns = [
        f"{metric}_{'correct' if correctness else 'wrong'}"
        for metric, correctness in pivot.columns
    ]

    pivot = pivot.reset_index()

    for col in [
        "count_correct",
        "count_wrong",
        "mean_value_correct",
        "mean_value_wrong",
        "max_value_correct",
        "max_value_wrong",
        "mean_rank_correct",
        "mean_rank_wrong",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["mean_value_wrong_minus_correct"] = (
        pivot["mean_value_wrong"] - pivot["mean_value_correct"]
    )

    pivot["rank_wrong_minus_correct"] = (
        pivot["mean_rank_wrong"] - pivot["mean_rank_correct"]
    )

    pivot = pivot.sort_values(
        "mean_value_wrong_minus_correct",
        ascending=False,
    )

    pivot.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved feature value analysis to: {OUTPUT_PATH}")
    print()

    print("Features with highest mean value in wrong cases relative to correct cases:")
    print(
        pivot[
            [
                "sae_feature_id",
                "count_wrong",
                "count_correct",
                "mean_value_wrong",
                "mean_value_correct",
                "mean_value_wrong_minus_correct",
                "mean_rank_wrong",
                "mean_rank_correct",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
