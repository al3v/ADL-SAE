from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVATION_METADATA_PATH = PROJECT_ROOT / "outputs" / "activations" / "activation_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layer1_top_features.csv"

REPO_ID = "google/gemma-scope-2b-pt-res"
SAE_PATH = "layer_1/width_16k/average_l0_20/params.npz"

LAYER_TO_USE = 1
TOP_K = 20


def load_sae_params() -> dict[str, torch.Tensor]:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=SAE_PATH,
    )

    data = np.load(local_path)

    params = {
        "W_enc": torch.tensor(data["W_enc"], dtype=torch.float32),
        "b_enc": torch.tensor(data["b_enc"], dtype=torch.float32),
        "b_dec": torch.tensor(data["b_dec"], dtype=torch.float32),
        "threshold": torch.tensor(data["threshold"], dtype=torch.float32),
    }

    print("Loaded SAE parameters:")
    for name, tensor in params.items():
        print(f"{name}: {tuple(tensor.shape)}")

    return params


def encode_activation(activation: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    activation = activation.float()

    centered = activation - params["b_dec"]
    pre_acts = centered @ params["W_enc"] + params["b_enc"]

    # Gemma Scope uses a thresholded JumpReLU-style SAE.
    features = pre_acts * (pre_acts > params["threshold"])

    return features


def main() -> None:
    metadata = pd.read_csv(ACTIVATION_METADATA_PATH)
    metadata = metadata[metadata["layer"] == LAYER_TO_USE].copy()

    params = load_sae_params()

    rows = []

    for _, row in metadata.iterrows():
        activation = torch.load(row["activation_path"])

        with torch.no_grad():
            features = encode_activation(activation, params)

        nonzero_count = int((features > 0).sum().item())
        feature_norm = torch.linalg.vector_norm(features).item()

        top_values, top_indices = torch.topk(features, k=TOP_K)

        for rank, (feature_id, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "question": row["question"],
                    "correct_answer": row["correct_answer"],
                    "model_answer": row["model_answer"],
                    "is_correct": row["is_correct"],
                    "type": row["type"],
                    "layer": row["layer"],
                    "sae_release": REPO_ID,
                    "sae_path": SAE_PATH,
                    "sae_feature_id": feature_id,
                    "sae_feature_value": value,
                    "feature_rank": rank,
                    "nonzero_feature_count": nonzero_count,
                    "feature_norm": feature_norm,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved SAE top features to: {OUTPUT_PATH}")
    print()

    print("Number of samples encoded:", metadata["sample_id"].nunique())
    print("Rows saved:", len(out_df))
    print()

    print("Average nonzero SAE features:")
    print(out_df.drop_duplicates("sample_id")["nonzero_feature_count"].mean())
    print()

    print("Average SAE feature norm by correctness:")
    print(
        out_df.drop_duplicates("sample_id")
        .groupby("is_correct")["feature_norm"]
        .mean()
    )
    print()

    print("Wrong cases top features:")
    wrong = out_df[out_df["is_correct"] == False]
    print(
        wrong[
            [
                "sample_id",
                "question",
                "correct_answer",
                "model_answer",
                "sae_feature_id",
                "sae_feature_value",
                "feature_rank",
                "nonzero_feature_count",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
