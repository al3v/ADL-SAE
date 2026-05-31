from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "model_answers.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "popqa_type_balanced.csv"

CANDIDATE_TYPES = [
    "screenwriter",
    "director",
    "producer",
    "author",
    "composer",
    "place of birth",
    "country",
    "father",
    "sport",
    "capital",
    "capital of",
]

N_PER_CLASS_PER_TYPE = 10
RANDOM_SEED = 42


def is_short_clean_answer(text: str) -> bool:
    text = str(text).strip()

    if not text:
        return False

    bad_patterns = [
        "Question:",
        "Answer:",
        "the father of the one",
        "is unknown",
    ]

    for pattern in bad_patterns:
        if pattern.lower() in text.lower():
            return False

    if len(text.split()) > 8:
        return False

    return True


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    df = df[df["type"].isin(CANDIDATE_TYPES)].copy()
    df = df[df["model_answer"].apply(is_short_clean_answer)].copy()

    selected_parts = []

    print("Available clean counts by type:")
    print(df.groupby(["type", "is_correct"]).size().unstack(fill_value=0))
    print()

    for qtype in CANDIDATE_TYPES:
        type_df = df[df["type"] == qtype].copy()

        correct = type_df[type_df["is_correct"] == True]
        wrong = type_df[type_df["is_correct"] == False]

        n = min(len(correct), len(wrong), N_PER_CLASS_PER_TYPE)

        if n == 0:
            print(f"{qtype}: skipped, not enough correct/wrong samples")
            continue

        correct_sample = correct.sample(n=n, random_state=RANDOM_SEED)
        wrong_sample = wrong.sample(n=n, random_state=RANDOM_SEED)

        selected_parts.append(correct_sample)
        selected_parts.append(wrong_sample)

        print(f"{qtype}: selected {n} correct + {n} wrong")

    if not selected_parts:
        raise RuntimeError("No balanced type subsets found.")

    balanced = pd.concat(selected_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    cols = ["question", "correct_answer", "possible_answers", "type"]
    balanced[cols].to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved type-balanced dataset to: {OUTPUT_PATH}")
    print("Rows:", len(balanced))
    print()
    print("Original label counts:")
    print(balanced["is_correct"].value_counts())
    print()
    print("Type counts:")
    print(balanced.groupby(["type", "is_correct"]).size())


if __name__ == "__main__":
    main()