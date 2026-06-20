import json
import os
import sys
import time
import argparse
import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv
    # Looks for a .env file in the project root (one level up from harness/)
    _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed — fall back to whatever is already in os.environ

ENDPOINT = "https://api-ap-southeast-1.modelarts-maas.com/anthropic/v1/messages"
MODELS = ["glm-5.1", "glm-5", "deepseek-v4-pro", "deepseek-v4-flash"]

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "master_dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 8
REQUEST_TIMEOUT = 120
RATE_LIMIT_SLEEP = 1.5  # gentle pacing between calls — increased to reduce 429s after observing heavy rate limiting


def get_api_key():
    key = os.environ.get("HUAWEI_API_KEY")
    if not key:
        print("ERROR: Set HUAWEI_API_KEY environment variable first.")
        print('  export HUAWEI_API_KEY="your-rotated-key-here"')
        sys.exit(1)
    return key


def call_model(model: str, prompt: str, api_key: str) -> dict:
    """Calls the Huawei ModelArts Anthropic-compatible endpoint for a single prompt."""
    body = {
        "model": model,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                text_blocks = [b["text"] for b in parsed.get("content", []) if b.get("type") == "text"]
                return {
                    "success": True,
                    "response_text": "\n".join(text_blocks),
                    "raw": parsed,
                }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {err_body}"
            if e.code == 429:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
            elif e.code >= 500:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            else:
                break
        except Exception as e:
            last_error = str(e)
            time.sleep(RETRY_DELAY_SECONDS)
            continue

    return {"success": False, "error": last_error, "response_text": None, "raw": None}


def load_dataset(limit=None, category_filter=None, language_filter=None):
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if category_filter:
        data = [d for d in data if d["category_id"] == category_filter]
    if language_filter:
        data = [d for d in data if d["language"] == language_filter]
    if limit:
        data = data[:limit]
    return data


def run_for_model(model: str, api_key: str, limit=None, category_filter=None, language_filter=None):
    dataset = load_dataset(limit, category_filter, language_filter)
    print(f"\n=== Running {len(dataset)} prompts against {model} ===")

    results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"raw_{model}.jsonl")

    # Resume support: skip IDs that already SUCCEEDED in a previous run.
    # Failed attempts are intentionally NOT added to done_ids, so a rerun
    # will retry them rather than silently treating a fail as final.
    done_ids = set()
    failed_count = 0
    if os.path.exists(out_path):
        kept_lines = []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("success"):
                    done_ids.add(rec["id"])
                    kept_lines.append(line)
                else:
                    failed_count += 1
                    # dropped: this failed record is not kept, so it will be retried below
        if done_ids or failed_count:
            print(f"Resuming: {len(done_ids)} prompts already succeeded for {model} (kept).")
        if failed_count:
            print(f"  {failed_count} previously-failed prompts will be RETRIED this run.")
        # Rewrite the file with only successful records; failed ones will be re-appended if they fail again
        with open(out_path, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")

    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(dataset, 1):
            if item["id"] in done_ids:
                continue

            result = call_model(model, item["prompt"], api_key)
            record = {
                "id": item["id"],
                "model": model,
                "category_id": item["category_id"],
                "category": item["category"],
                "language": item["language"],
                "language_full": item["language_full"],
                "severity": item["severity"],
                "subcategory": item["subcategory"],
                "prompt": item["prompt"],
                "success": result["success"],
                "response_text": result.get("response_text"),
                "error": result.get("error"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            results.append(record)

            status = "OK" if result["success"] else f"FAIL ({result.get('error', '')[:80]})"
            print(f"[{i}/{len(dataset)}] {item['id']} ({item['language']}/{item['category_id']}) -> {status}")
            time.sleep(RATE_LIMIT_SLEEP)

    print(f"\nDone. Results appended to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="AfriSafe-Eval harness")
    parser.add_argument("--model", required=True, help="Model name, or 'all' for all 4 models")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts (for testing)")
    parser.add_argument("--category", default=None, help="Filter by category_id, e.g. CAT1")
    parser.add_argument("--language", default=None, help="Filter by language code, e.g. zu")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.model == "all":
        for m in MODELS:
            run_for_model(m, api_key, args.limit, args.category, args.language)
    else:
        if args.model not in MODELS:
            print(f"WARNING: '{args.model}' not in known model list {MODELS}. Proceeding anyway.")
        run_for_model(args.model, api_key, args.limit, args.category, args.language)


if __name__ == "__main__":
    main()
