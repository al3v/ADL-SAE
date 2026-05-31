from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZATION_PATH = PROJECT_ROOT / "outputs" / "tokenization_analysis.csv"
ACTIVATION_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "activation_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "merged_tokenization_activation.csv"


def main() -> None:
    tok_df = pd.read_csv(TOKENIZATION_PATH)
    act_df = pd.read_csv(ACTIVATION_SUMMARY_PATH)

    tok_df = tok_df.reset_index().rename(columns={"index": "sample_id"})

    merged = act_df.merge(
        tok_df[
            [
                "sample_id",
                "correct_answer_num_tokens",
                "model_answer_num_tokens",
                "is_correct_answer_single_token",
                "is_model_answer_single_token",
                "correct_answer_tokens",
                "model_answer_tokens",
            ]
        ],
        on="sample_id",
        how="left",
    )

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved merged analysis to: {OUTPUT_PATH}")
    print()

    print("Average activation norm by layer:")
    print(merged.groupby("layer")["activation_norm"].mean())
    print()

    print("Average activation norm by layer and correct-answer tokenization:")
    print(
        merged.groupby(["layer", "is_correct_answer_single_token"])["activation_norm"]
        .mean()
        .reset_index()
    )
    print()

    print("Average activation norm by layer and answer correctness:")
    print(
        merged.groupby(["layer", "is_correct"])["activation_norm"]
        .mean()
        .reset_index()
    )


if __name__ == "__main__":
    main()
