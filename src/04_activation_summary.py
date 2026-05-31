from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "outputs" / "activations" / "activation_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "activation_summary.csv"


def main() -> None:
    df = pd.read_csv(METADATA_PATH)

    rows = []

    for _, row in df.iterrows():
        activation = torch.load(row["activation_path"])

        rows.append(
            {
                "sample_id": row["sample_id"],
                "question": row["question"],
                "correct_answer": row["correct_answer"],
                "model_answer": row["model_answer"],
                "is_correct": row["is_correct"],
                "type": row["type"],
                "layer": row["layer"],
                "activation_mean": activation.float().mean().item(),
                "activation_std": activation.float().std().item(),
                "activation_norm": torch.linalg.vector_norm(activation.float()).item(),
                "activation_max": activation.float().max().item(),
                "activation_min": activation.float().min().item(),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved activation summary to: {OUTPUT_PATH}")
    print(out_df.head())

    print("\nLayer-wise average activation norm:")
    print(out_df.groupby("layer")["activation_norm"].mean())


if __name__ == "__main__":
    main()
