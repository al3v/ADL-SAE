from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layers_6_9_12_top100_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sae_layers_6_9_12_top100_feature_by_type.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer].copy()

        for qtype in sorted(layer_df["type"].unique()):
            type_df = layer_df[layer_df["type"] == qtype].copy()

            num_correct = type_df[type_df["is_correct"] == True]["sample_id"].nunique()
            num_wrong = type_df[type_df["is_correct"] == False]["sample_id"].nunique()

            if num_correct == 0 or num_wrong == 0:
                continue

            grouped = (
                type_df.groupby(["sae_feature_id", "is_correct"])
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
            pivot["type"] = qtype
            pivot["num_correct_samples_in_type"] = num_correct
            pivot["num_wrong_samples_in_type"] = num_wrong
            pivot["freq_correct_within_type"] = pivot["count_correct"] / num_correct
            pivot["freq_wrong_within_type"] = pivot["count_wrong"] / num_wrong
            pivot["freq_diff_wrong_minus_correct_within_type"] = (
                pivot["freq_wrong_within_type"] - pivot["freq_correct_within_type"]
            )

            rows.append(pivot)

    out_df = pd.concat(rows, ignore_index=True)
    out_df = out_df.sort_values(
        ["layer", "freq_diff_wrong_minus_correct_within_type"],
        ascending=[True, False],
    )

    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved later-layer type-controlled analysis to: {OUTPUT_PATH}")
    print()

    for layer in sorted(out_df["layer"].unique()):
        print("=" * 80)
        print(f"Layer {layer}")
        print("=" * 80)

        layer_out = out_df[out_df["layer"] == layer]

        print(
            layer_out[
                [
                    "type",
                    "sae_feature_id",
                    "count_wrong",
                    "count_correct",
                    "freq_wrong_within_type",
                    "freq_correct_within_type",
                    "freq_diff_wrong_minus_correct_within_type",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )
        print()


if __name__ == "__main__":
    main()