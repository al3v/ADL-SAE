from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "model_answers.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "popqa_clean_candidate_subset.csv"

KEEP_TYPES = {
    "screenwriter",
    "composer",
    "director",
    "producer",
    "author",
    "place of birth",
    "father",
}

REMOVE_MODEL_ANSWER_PATTERNS = [
    "Question:",
    "the father of the one who is the father",
]


def looks_clean_model_answer(text: str) -> bool:
    text = str(text)

    if len(text.split()) > 8:
        return False

    for pattern in REMOVE_MODEL_ANSWER_PATTERNS:
        if pattern.lower() in text.lower():
            return False

    return True


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    clean = df[df["type"].isin(KEEP_TYPES)].copy()
    clean = clean[clean["model_answer"].apply(looks_clean_model_answer)].copy()

    clean.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved clean candidate subset to: {OUTPUT_PATH}")
    print()
    print("Rows:", len(clean))
    print()
    print("Correctness counts:")
    print(clean["is_correct"].value_counts())
    print()
    print("Counts by type and correctness:")
    print(clean.groupby(["type", "is_correct"]).size())
    print()
    print("First 50 wrong examples:")
    wrong = clean[clean["is_correct"] == False][
        ["question", "correct_answer", "model_answer", "type"]
    ]

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 220)
    print(wrong.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
