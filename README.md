# AfriSafe-Eval — Working Files (Hackathon Build)

## Status: All 4 models complete (1,600 calls) · Classifier fully validated (n=40)

## What this is
A red-teaming dataset and evaluation harness for testing LLM safety failures
in South African languages (English, isiZulu, isiXhosa, Sesotho, Afrikaans)
across four harm categories grounded in real South African deployment risks:

1. Electoral manipulation (CAT1)
2. Healthcare misinformation (CAT2)
3. Financial fraud (CAT3)
4. Gender-based violence facilitation (CAT4)

Built for the Apart Research Global South AI Safety Hackathon (19–21 June 2026).

## Headline findings (n=1,600 · 4 models · 5 languages)

- **Overall harmful response rate: 38.3%** (612 / 1,600 responses complied or partially complied)
- **isiXhosa has the highest harmful response rate (61.2%)**, with a 43.4 pp gap vs. the safest
  language, Afrikaans (17.8%).
- **The Nguni divergence**: isiXhosa (61.2%) and isiZulu (23.8%) are both Nguni languages with
  broadly comparable training-data representation, yet differ by 37 pp — suggesting
  language-specific alignment failures beyond simple resource availability.
- **Electoral manipulation is the highest-risk category (51.2%)**, followed by healthcare
  misinformation (41.8%), financial fraud (35.2%), and GBV facilitation (24.8%).
- **Medium-severity prompts bypassed safety more often (53.5%) than high-severity (33.2%).**
  Oblique framing is more effective at eliciting harmful completions than direct requests.

See `figures/heatmap_final.png` and `figures/model_comparison.png` for visual breakdowns.

### Harmful response rate by model

| Model             | Harmful rate | COMPLY | PARTIAL | REFUSE | CORRECTIVE | BLOCKED |
|-------------------|:-----------:|:------:|:-------:|:------:|:----------:|:-------:|
| glm-5             | 34.2%       | 134    | 3       | 221    | 3          | 36      |
| glm-5.1           | 41.8%       | 150    | 17      | 198    | 0          | 33      |
| deepseek-v4-flash | 40.5%       | 147    | 15      | 204    | 9          | 25      |
| deepseek-v4-pro   | 36.5%       | 133    | 13      | 209    | 12         | 31      |

*BLOCKED = provider-level filter caught the prompt before it reached the model.*

### Harmful response rate by language

| Language  | Harmful rate | n   |
|-----------|:-----------:|:---:|
| isiXhosa  | 61.2%       | 320 |
| Sesotho   | 45.0%       | 320 |
| English   | 43.4%       | 320 |
| isiZulu   | 23.8%       | 320 |
| Afrikaans | 17.8%       | 320 |

## Classifier validation

The automated classifier (rule-based refusal-pattern matching in `harness/score_results.py`)
was validated against 40 human-labelled items from a stratified random sample.

**Note on normalisation**: `PROVIDER_BLOCKED` and `CORRECTIVE_REFUSAL` auto-labels are
normalised to `REFUSE` before computing metrics. This aligns the auto-label space with
the human annotation schema (REFUSE / COMPLY / PARTIAL / CORRECTIVE_REFUSAL).

| Metric                     | Value  |
|----------------------------|:------:|
| n labelled                 | 40     |
| Multi-class accuracy       | 72.5%  |
| Binary accuracy (harmful vs not) | 92.5% |
| Not-harmful F1             | 0.947  |
| Harmful F1                 | 0.870  |
| ROC AUC (proxy score)      | 0.617  |

The multi-class accuracy (72.5%) is lower than binary accuracy (92.5%) primarily because
the classifier cannot distinguish `CORRECTIVE_REFUSAL` from `REFUSE` (it has no
corrective-pattern output label in the auto-classification path). For safety-research
purposes — where the key question is "did the model produce harmful content?" — the
binary metric (92.5%) is the more meaningful number.

Remaining known weakness: the classifier misses subtle in-language refusals in isiXhosa
and Sesotho that don't trigger the keyword lists, causing false COMPLY labels.
See `results/manual_validation_sample.json` for the 3 documented mismatches.

## IMPORTANT — Handle with care
The `dataset/` folder contains 400 deliberately harmful prompts (deepfake scripts,
scam scripts, health disinformation, GBV facilitation requests). This is standard
red-teaming methodology — the same approach used in Apart Research's own Darkbench
benchmark and the academic literature on African-language jailbreaking. The prompts
exist to measure whether models refuse them, not to be used for their literal purpose.

The `results/raw_*.jsonl` files contain real harmful completions from live models.
Treat as sensitive working data — do not post these raw files publicly. Only the
aggregated statistics and a small number of redacted examples should go in the
public paper/repo.

## Folder structure
```
dataset/    20 language/category JSON files + master_dataset.json (400 prompts merged)
harness/    build_master_dataset.py, run_eval.py, score_results.py, validate_classifier.py
results/    raw_*.jsonl (4 models, 1600 calls total), scored_results.jsonl,
            manual_validation_sample.json (40 items — fully labelled),
            classifier_validation_summary.json
figures/    heatmap_final.png, model_comparison.png, heatmap_lang_x_category.png,
            confusion_matrix.png, roc_curve.png
paper/      (for the research writeup — not yet drafted)
```

## How to run (do this on your own machine)
```bash
# 1. Set up environment
pip install -r requirements.txt
cp .env .env.example   # .env already has placeholder — fill in your real key
export HUAWEI_API_KEY="your-key-here"

# 2. (Already done) Run the eval — all 4 models, 1600 calls total
cd harness
python3 run_eval.py --model deepseek-v4-flash
python3 run_eval.py --model deepseek-v4-pro
python3 run_eval.py --model glm-5.1
python3 run_eval.py --model glm-5

# 3. (Already done) Score the results
python3 score_results.py

# 4. (Already done) Run classifier validation
python3 validate_classifier.py
# Produces: figures/confusion_matrix.png, figures/roc_curve.png,
#           results/classifier_validation_summary.json
```


