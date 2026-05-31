from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


REPO_ID = "google/gemma-scope-2b-pt-res"
SAE_PATH = "layer_1/width_16k/average_l0_20/params.npz"


def main() -> None:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=SAE_PATH,
    )

    print("Downloaded to:")
    print(local_path)
    print()

    data = np.load(local_path)

    print("Keys and shapes:")
    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
