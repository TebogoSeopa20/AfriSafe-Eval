import json
import glob
import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

def build_master():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "cat*_*.json")))
    master = []
    seen_ids = set()

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        category_id = data["category_id"]
        category = data["category"]
        language = data["language"]
        language_full = data["language_full"]

        for p in data["prompts"]:
            entry = {
                "id": p["id"],
                "category_id": category_id,
                "category": category,
                "language": language,
                "language_full": language_full,
                "severity": p["severity"],
                "subcategory": p["subcategory"],
                "prompt": p["prompt"],
            }
            if entry["id"] in seen_ids:
                raise ValueError(f"Duplicate ID found: {entry['id']}")
            seen_ids.add(entry["id"])
            master.append(entry)

    out_path = os.path.join(DATASET_DIR, "master_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)

    print(f"Built master dataset: {len(master)} prompts -> {out_path}")

    # Summary stats
    from collections import Counter
    cat_counts = Counter(e["category_id"] for e in master)
    lang_counts = Counter(e["language"] for e in master)
    sev_counts = Counter(e["severity"] for e in master)

    print("\nBy category:", dict(cat_counts))
    print("By language:", dict(lang_counts))
    print("By severity:", dict(sev_counts))

if __name__ == "__main__":
    build_master()
