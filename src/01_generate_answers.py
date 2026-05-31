from pathlib import Path
import ast
import json
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

    # Normalize common punctuation and possessive apostrophes.
    text = text.replace("’", "'")

    # Remove punctuation but keep letters/numbers/spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Collapse separated initials:
    # "j j thomson" -> "jj thomson"
    text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text)

    return text


def parse_possible_answers(value, fallback_answer: str) -> list[str]:
    answers = []

    if value is not None and not pd.isna(value):
        if isinstance(value, list):
            answers = value
        elif isinstance(value, str):
            text = value.strip()

            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        answers = parsed
                except Exception:
                    try:
                        parsed = ast.literal_eval(text)
                        if isinstance(parsed, list):
                            answers = parsed
                    except Exception:
                        answers = [text]
            elif text:
                answers = [text]

    if not answers:
        answers = [fallback_answer]

    # Always include fallback correct_answer too.
    answers.append(fallback_answer)

    cleaned = []
    seen = set()

    for answer in answers:
        answer_text = str(answer).strip()

        if not answer_text or answer_text.lower() == "nan":
            continue

        key = normalize_text(answer_text)

        if key and key not in seen:
            cleaned.append(answer_text)
            seen.add(key)

    return cleaned


def is_answer_correct(model_answer: str, possible_answers: list[str]) -> bool:
    model_answer_norm = normalize_text(model_answer)

    for answer in possible_answers:
        answer_norm = normalize_text(answer)

        if not answer_norm:
            continue

        # Normal exact/substring match.
        if answer_norm in model_answer_norm:
            return True

        # Accept surname-only answers for named people:
        # "Werner Heisenberg" -> "Heisenberg"
        parts = answer_norm.split()
        if len(parts) >= 2:
            surname = parts[-1]
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

    model.eval()
    model.config.use_cache = False

    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, "cache_implementation"):
            model.generation_config.cache_implementation = None

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        correct_answer = row["correct_answer"]

        possible_answers = parse_possible_answers(
            row.get("possible_answers", None),
            correct_answer,
        )

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "Answer:" in decoded:
            model_answer = decoded.split("Answer:", 1)[-1].strip()
        else:
            model_answer = decoded.strip()

        model_answer = model_answer.split("\n")[0].strip()

        correct = is_answer_correct(model_answer, possible_answers)

        results.append(
            {
                "question": question,
                "correct_answer": correct_answer,
                "possible_answers": json.dumps(possible_answers, ensure_ascii=False),
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