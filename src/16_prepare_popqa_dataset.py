from pathlib import Path
import ast
import json

import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_SAMPLE_PATH = PROJECT_ROOT / "data" / "popqa_sample_with_aliases.csv"
OUTPUT_FULL_PATH = PROJECT_ROOT / "data" / "popqa_full_with_aliases.csv"

DATASET_NAME = "akariasai/PopQA"
N_SAMPLES = 2000
RANDOM_SEED = 42


def parse_answer_list(value) -> list[str]:
    """
    PopQA may store possible answers as:
    - a real Python list
    - a string that looks like a list: '["A", "B"]'
    - a plain string

    We return a cleaned list of answer aliases.
    """
    if value is None:
        return []

    if isinstance(value, list):
        raw_answers = value
    elif isinstance(value, str):
        text = value.strip()

        if not text or text.lower() == "nan":
            return []

        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    raw_answers = parsed
                else:
                    raw_answers = [text]
            except Exception:
                raw_answers = [text]
        else:
            raw_answers = [text]
    else:
        raw_answers = [value]

    cleaned = []
    seen = set()

    for answer in raw_answers:
        answer_text = str(answer).strip()

        if not answer_text or answer_text.lower() == "nan":
            continue

        key = answer_text.lower()

        if key not in seen:
            cleaned.append(answer_text)
            seen.add(key)

    return cleaned


def get_possible_answers(row_dict: dict) -> list[str]:
    for col in ["possible_answers", "answers", "answer", "obj"]:
        if col in row_dict:
            answers = parse_answer_list(row_dict[col])
            if answers:
                return answers

    return []


def get_type_from_row(row_dict: dict) -> str:
    if "prop" in row_dict and str(row_dict["prop"]).strip():
        return str(row_dict["prop"]).strip()

    if "relation" in row_dict and str(row_dict["relation"]).strip():
        return str(row_dict["relation"]).strip()

    return "popqa"


def main() -> None:
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    print(dataset)

    if "train" in dataset:
        df = dataset["train"].to_pandas()
    else:
        split_name = list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()

    print("\nColumns:")
    print(df.columns.tolist())
    print("\nNumber of rows:", len(df))

    if "question" not in df.columns:
        raise ValueError("No question column found.")

    records = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        question = str(row_dict.get("question", "")).strip()
        possible_answers = get_possible_answers(row_dict)
        qtype = get_type_from_row(row_dict)

        if not question or not possible_answers:
            continue

        records.append(
            {
                "question": question,
                "correct_answer": possible_answers[0],
                "possible_answers": json.dumps(possible_answers, ensure_ascii=False),
                "type": qtype,
            }
        )

    clean_df = pd.DataFrame(records)

    OUTPUT_FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUTPUT_FULL_PATH, index=False)

    sample_df = clean_df.sample(
        n=min(N_SAMPLES, len(clean_df)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    sample_df.to_csv(OUTPUT_SAMPLE_PATH, index=False)

    print("\nSaved full PopQA dataset to:", OUTPUT_FULL_PATH)
    print("Full rows:", len(clean_df))

    print("\nSaved PopQA sample to:", OUTPUT_SAMPLE_PATH)
    print("Sample rows:", len(sample_df))
    print()
    print(sample_df.head(20))


if __name__ == "__main__":
    main()
