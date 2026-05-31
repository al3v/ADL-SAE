from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVATION_METADATA_PATH = (
    PROJECT_ROOT / "outputs" / "activations_later" / "activation_metadata.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "minimal_sae_layers_6_9_12_top100_features.csv"

REPO_ID = "google/gemma-scope-2b-pt-res"

LAYERS_TO_USE = [6, 9, 12]
TARGET_L0 = 30
TOP_K = 100


def find_sae_path(layer: int) -> str:
    files = list_repo_files(REPO_ID)

    candidates = []
    prefix = f"layer_{layer}/width_16k/average_l0_"

    for file_name in files:
        if file_name.startswith(prefix) and file_name.endswith("params.npz"):
            match = re.search(r"average_l0_(\d+)", file_name)
            if match:
                l0_value = int(match.group(1))
                candidates.append((abs(l0_value - TARGET_L0), l0_value, file_name))

    if not candidates:
        raise ValueError(f"No width_16k SAE found for layer {layer}")

    candidates.sort()
    _, l0_value, selected_path = candidates[0]

    print(f"Selected SAE for layer {layer}: {selected_path} (average_l0={l0_value})")

    return selected_path


def load_sae_params(sae_path: str) -> dict[str, torch.Tensor]:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=sae_path,
    )

    data = np.load(local_path)

    params = {
        "W_enc": torch.tensor(data["W_enc"], dtype=torch.float32),
        "b_enc": torch.tensor(data["b_enc"], dtype=torch.float32),
        "b_dec": torch.tensor(data["b_dec"], dtype=torch.float32),
        "threshold": torch.tensor(data["threshold"], dtype=torch.float32),
    }

    print(f"Loaded SAE: {sae_path}")
    for name, tensor in params.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    return params


def encode_activation(
    activation: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    activation = activation.float()

    centered = activation - params["b_dec"]
    pre_acts = centered @ params["W_enc"] + params["b_enc"]

    # Gemma Scope JumpReLU-style thresholded features.
    features = pre_acts * (pre_acts > params["threshold"])

    return features


def main() -> None:
    metadata = pd.read_csv(ACTIVATION_METADATA_PATH)
    metadata = metadata[metadata["layer"].isin(LAYERS_TO_USE)].copy()

    sae_paths = {layer: find_sae_path(layer) for layer in LAYERS_TO_USE}

    all_rows = []

    for layer in LAYERS_TO_USE:
        sae_path = sae_paths[layer]

        print()
        print(f"Processing layer {layer}")
        print("-" * 40)

        layer_metadata = metadata[metadata["layer"] == layer].copy()
        params = load_sae_params(sae_path)

        for _, row in layer_metadata.iterrows():
            activation = torch.load(row["activation_path"])

            with torch.no_grad():
                features = encode_activation(activation, params)

            nonzero_count = int((features > 0).sum().item())
            feature_norm = torch.linalg.vector_norm(features).item()

            top_values, top_indices = torch.topk(features, k=TOP_K)

            for rank, (feature_id, value) in enumerate(
                zip(top_indices.tolist(), top_values.tolist()),
                start=1,
            ):
                all_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "question": row["question"],
                        "correct_answer": row["correct_answer"],
                        "possible_answers": row.get("possible_answers", ""),
                        "model_answer": row["model_answer"],
                        "is_correct": row["is_correct"],
                        "type": row["type"],
                        "layer": layer,
                        "sae_release": REPO_ID,
                        "sae_path": sae_path,
                        "sae_feature_id": feature_id,
                        "sae_feature_value": value,
                        "feature_rank": rank,
                        "nonzero_feature_count": nonzero_count,
                        "feature_norm": feature_norm,
                    }
                )

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved later-layer SAE top features to: {OUTPUT_PATH}")
    print("Rows saved:", len(out_df))
    print("Samples:", out_df["sample_id"].nunique())
    print("Layers:", sorted(out_df["layer"].unique()))
    print()

    sample_level = out_df.drop_duplicates(["sample_id", "layer"])

    print("Average nonzero SAE features by layer:")
    print(sample_level.groupby("layer")["nonzero_feature_count"].mean())
    print()

    print("Average nonzero SAE features by layer and correctness:")
    print(sample_level.groupby(["layer", "is_correct"])["nonzero_feature_count"].mean())
    print()

    print("Average SAE feature norm by layer and correctness:")
    print(sample_level.groupby(["layer", "is_correct"])["feature_norm"].mean())


if __name__ == "__main__":
    main()