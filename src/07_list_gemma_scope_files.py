from huggingface_hub import list_repo_files

REPO_ID = "google/gemma-scope-2b-pt-res"

def main() -> None:
    files = list_repo_files(REPO_ID)

    print(f"Total files: {len(files)}")
    print()

    layer_1_files = [f for f in files if "layer_1" in f.lower() or "layer1" in f.lower()]

    print("Files containing layer_1 / layer1:")
    for f in layer_1_files[:100]:
        print(f)

    print()
    print("First 100 files overall:")
    for f in files[:100]:
        print(f)


if __name__ == "__main__":
    main()
