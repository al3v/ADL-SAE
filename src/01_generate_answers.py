from pathlib import Path

import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "factual_questions.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "model_answers.csv"

MODEL_NAME = "google/gemma-2-2b"


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Collapse separated initials:
    # "j j thomson" -> "jj thomson"
    text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text)

    return text

def is_answer_correct(model_answer: str, correct_answer: str) -> bool:
    model_answer_norm = normalize_text(model_answer)
    correct_answer_norm = normalize_text(correct_answer)

    if correct_answer_norm in model_answer_norm:
        return True

    # Accept surname-only answers for named people:
    # "Werner Heisenberg" -> "Heisenberg"
    correct_parts = correct_answer_norm.split()
    if len(correct_parts) >= 2:
        surname = correct_parts[-1]
        if len(surname) > 4 and surname in model_answer_norm:
            return True

    return False
def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        correct_answer = row["correct_answer"]

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                use_cache=False,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "Answer:" in decoded:
            model_answer = decoded.split("Answer:", 1)[-1].strip()
        else:
            model_answer = decoded.strip()

        model_answer = model_answer.split("\n")[0].strip()

        correct = is_answer_correct(model_answer, correct_answer)

        results.append(
            {
                "question": question,
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "is_correct": correct,
                "type": row["type"],
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(out_df)


if __name__ == "__main__":
    main()
