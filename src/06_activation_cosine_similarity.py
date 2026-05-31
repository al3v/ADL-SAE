from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "outputs" / "activations" / "activation_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "activation_cosine_similarity.csv"


def load_activation(path: str) -> torch.Tensor:
    activation = torch.load(path)
    return activation.float()


def main() -> None:
    df = pd.read_csv(METADATA_PATH)

    rows = []

    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer].copy().reset_index(drop=True)

        activations = []
        for _, row in layer_df.iterrows():
            act = load_activation(row["activation_path"])
            activations.append(act)

        activation_matrix = torch.stack(activations, dim=0)
        activation_matrix = F.normalize(activation_matrix, dim=1)

        similarity_matrix = activation_matrix @ activation_matrix.T

        for i, row_i in layer_df.iterrows():
            similarities = similarity_matrix[i]

            # remove self-similarity
            mask_not_self = torch.ones(len(layer_df), dtype=torch.bool)
            mask_not_self[i] = False

            same_correctness_mask = (
                (layer_df["is_correct"] == row_i["is_correct"]).to_numpy()
            )
            different_correctness_mask = (
                (layer_df["is_correct"] != row_i["is_correct"]).to_numpy()
            )
            same_type_mask = (layer_df["type"] == row_i["type"]).to_numpy()

            same_correctness_mask = torch.tensor(same_correctness_mask) & mask_not_self
            different_correctness_mask = torch.tensor(different_correctness_mask) & mask_not_self
            same_type_mask = torch.tensor(same_type_mask) & mask_not_self

            def safe_mean(mask: torch.Tensor):
                if mask.sum().item() == 0:
                    return None
                return similarities[mask].mean().item()

            rows.append(
                {
                    "sample_id": row_i["sample_id"],
                    "question": row_i["question"],
                    "correct_answer": row_i["correct_answer"],
                    "model_answer": row_i["model_answer"],
                    "is_correct": row_i["is_correct"],
                    "type": row_i["type"],
                    "layer": layer,
                    "mean_similarity_same_correctness": safe_mean(same_correctness_mask),
                    "mean_similarity_different_correctness": safe_mean(different_correctness_mask),
                    "mean_similarity_same_type": safe_mean(same_type_mask),
                    "mean_similarity_all_others": similarities[mask_not_self].mean().item(),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cosine similarity analysis to: {OUTPUT_PATH}")
    print()

    print("Mean similarity by layer:")
    print(out_df.groupby("layer")["mean_similarity_all_others"].mean())
    print()

    print("Mean similarity by layer and correctness:")
    print(
        out_df.groupby(["layer", "is_correct"])[
            "mean_similarity_all_others"
        ].mean()
    )
    print()

    print("Wrong cases:")
    wrong = out_df[out_df["is_correct"] == False]
    print(
        wrong[
            [
                "layer",
                "question",
                "correct_answer",
                "model_answer",
                "mean_similarity_same_type",
                "mean_similarity_all_others",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
