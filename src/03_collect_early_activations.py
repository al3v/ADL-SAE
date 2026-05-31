from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "model_answers.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "activations"

MODEL_NAME = "google/gemma-2-2b"
LAYERS_TO_SAVE = [1, 2, 3]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = False

    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        prompt = f"Question: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states

        for layer_idx in LAYERS_TO_SAVE:
            activation = hidden_states[layer_idx][0, -1, :].detach().cpu()

            save_path = OUTPUT_DIR / f"sample_{idx}_layer_{layer_idx}.pt"
            torch.save(activation, save_path)

            records.append(
                {
                    "sample_id": idx,
                    "question": question,
                    "correct_answer": row["correct_answer"],
                    "model_answer": row["model_answer"],
                    "is_correct": row["is_correct"],
                    "type": row["type"],
                    "layer": layer_idx,
                    "activation_path": str(save_path),
                    "activation_shape": list(activation.shape),
                }
            )

    metadata_df = pd.DataFrame(records)
    metadata_path = OUTPUT_DIR / "activation_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"Saved activation metadata to: {metadata_path}")
    print(metadata_df.head())


if __name__ == "__main__":
    main()
