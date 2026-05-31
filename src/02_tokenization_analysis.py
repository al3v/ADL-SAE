from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "model_answers.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tokenization_analysis.csv"

MODEL_NAME = "google/gemma-2-2b"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows = []

    for _, row in df.iterrows():
        correct_answer = str(row["correct_answer"])
        model_answer = str(row["model_answer"])

        correct_answer_token_ids = tokenizer.encode(
            correct_answer,
            add_special_tokens=False,
        )
        model_answer_token_ids = tokenizer.encode(
            model_answer,
            add_special_tokens=False,
        )

        correct_answer_tokens = tokenizer.convert_ids_to_tokens(correct_answer_token_ids)
        model_answer_tokens = tokenizer.convert_ids_to_tokens(model_answer_token_ids)

        rows.append(
            {
                "question": row["question"],
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "is_correct": row["is_correct"],
                "type": row["type"],
                "correct_answer_num_tokens": len(correct_answer_token_ids),
                "model_answer_num_tokens": len(model_answer_token_ids),
                "correct_answer_tokens": str(correct_answer_tokens),
                "model_answer_tokens": str(model_answer_tokens),
                "is_correct_answer_single_token": len(correct_answer_token_ids) == 1,
                "is_model_answer_single_token": len(model_answer_token_ids) == 1,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved tokenization analysis to: {OUTPUT_PATH}")
    print(out_df)


if __name__ == "__main__":
    main()
