from pathlib import Path
import ast

import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "popqa_sample.csv"

DATASET_NAME = "akariasai/PopQA"
N_SAMPLES = 300
RANDOM_SEED = 42


def extract_first_answer(value) -> str | None:
    """
    PopQA may store answers as:
    - a real Python list
    - a string that looks like a list: '["A", "B"]'
    - a plain string
    We always return the first usable answer.
    """
    if value is None:
        return None

    if isinstance(value, list):
        if len(value) == 0:
            return None
        return str(value[0]).strip()

    if isinstance(value, str):
        text = value.strip()

        if not text or text.lower() == "nan":
            return None

        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return str(parsed[0]).strip()
            except Exception:
                pass

        return text

    return str(value).strip()


def get_answer_from_row(row_dict: dict) -> str | None:
    """
    Try common PopQA answer columns.
    """
    possible_answer_columns = [
        "possible_answers",
        "answers",
        "answer",
        "obj",
    ]

    for col in possible_answer_columns:
        if col in row_dict:
            answer = extract_first_answer(row_dict[col])
            if answer:
                return answer

    return None


def get_type_from_row(row_dict: dict) -> str:
    """
    Use relation/property as the type if available.
    """
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
    print("\nFirst rows:")
    print(df.head())

    if "question" not in df.columns:
        raise ValueError("No question column found.")

    records = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        question = str(row_dict.get("question", "")).strip()
        answer = get_answer_from_row(row_dict)
        qtype = get_type_from_row(row_dict)

        if not question or not answer:
            continue

        records.append(
            {
                "question": question,
                "correct_answer": answer,
                "type": qtype,
            }
        )

    clean_df = pd.DataFrame(records)

    print("\nClean rows:", len(clean_df))
    print(clean_df.head())

    sample_df = clean_df.sample(
        n=min(N_SAMPLES, len(clean_df)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved PopQA sample to: {OUTPUT_PATH}")
    print("Rows:", len(sample_df))
    print(sample_df.head(20))


if __name__ == "__main__":
    main()