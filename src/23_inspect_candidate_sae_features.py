from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layers_6_9_12_top100_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "candidate_sae_feature_examples.csv"

LAYER = 12
FEATURE_IDS = [13338, 3830, 8649, 4820, 2987]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for feature_id in FEATURE_IDS:
        hits = df[
            (df["layer"] == LAYER)
            & (df["sae_feature_id"] == feature_id)
        ].copy()

        print("=" * 100)
        print(f"Layer {LAYER}, SAE feature {feature_id}")
        print("=" * 100)

        print("Counts by correctness:")
        print(hits["is_correct"].value_counts())
        print()

        print("Counts by type and correctness:")
        print(hits.groupby(["type", "is_correct"]).size())
        print()

        print("Top examples by feature value:")
        view = hits.sort_values("sae_feature_value", ascending=False)[
            [
                "sample_id",
                "question",
                "correct_answer",
                "model_answer",
                "is_correct",
                "type",
                "sae_feature_value",
                "feature_rank",
            ]
        ]

        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", 260)

        print(view.head(30).to_string(index=False))
        print()

        rows.append(hits)

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved candidate feature examples to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()