# LightMem

A memory management framework for LLMs, using LLMLingua-2 for prompt compression, Qdrant as a local vector store, and HuggingFace sentence-transformers for embeddings.

## Installation

```bash
pip install -e .
```

Requires Python 3.10 or 3.11.

For AWS Bedrock support:
```bash
pip install boto3
```

## Experiments

All experiments follow a two-step pattern: **add** (build memory) → **search** (retrieve & evaluate).

Set your API credentials as environment variables before running:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://api.openai.com/v1   # or your custom endpoint
```

---

### LoCoMo

```bash
cd experiments/locomo

# Build memory
python add_locomo.py --workers 5

# Evaluate
python search_locomo.py \
  --dataset ../../data/locomo/locomo10.json \
  --qdrant-dir ./qdrant_post_update \
  --llm-api-key $OPENAI_API_KEY \
  --llm-base-url $OPENAI_BASE_URL \
  --llm-model gpt-4o-mini \
  --judge-api-key $OPENAI_API_KEY \
  --judge-base-url $OPENAI_BASE_URL \
  --judge-model gpt-4o-mini \
  --output-dir ./results
```

---

### HotpotQA

```bash
cd experiments/hotpotqa

# Build memory (batched, ~3hrs for all 1600 samples)
python add_hotpotqa.py --limit 50 --batch-docs 20

# Evaluate
python search_hotpotqa.py \
  --qdrant-dir ./qdrant_post_update \
  --llm-api-key $OPENAI_API_KEY \
  --llm-base-url $OPENAI_BASE_URL \
  --llm-model gpt-4o-mini \
  --output-dir ./results_hotpotqa
```

---

### LongMemEval

```bash
cd experiments/longmemeval

# OpenAI-compatible endpoint
python run_lightmem_gpt.py

# Qwen / custom endpoint (set TINKER_API_KEY and TINKER_MODEL env vars)
python run_lightmem_qwen.py
```

---

### RULER 128k

```bash
cd experiments/ruler

# Build memory
python add_ruler128k.py

# Evaluate
python search_ruler128k.py \
  --qdrant-dir ./qdrant_post_update \
  --output-dir ./results_ruler128k
```

---

### Haiku (AWS Bedrock)

Uses Claude Haiku via AWS Bedrock instead of an OpenAI-compatible endpoint.

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
export AWS_INFERENCE_PROFILE_ID=your_profile_id

cd experiments/haiku

# Build memory
python add_haiku.py --workers 5

# Evaluate
python search_haiku.py \
  --dataset ../../data/locomo/locomo10.json \
  --qdrant-dir ./qdrant_post_update \
  --output-dir ./haiku_results
```

---

## Common Arguments

| Argument | Description |
|---|---|
| `--limit N` | Process only first N samples |
| `--workers N` | Parallel workers (default: 5) |
| `--device cpu\|cuda` | Device for embedding model |
| `--extraction_mode flat\|event` | Memory extraction mode |
| `--enable_summary` | Enable session-level summarization |
| `--output-dir PATH` | Where to save results |

## Output

Each experiment saves results to its output directory:
- `batch_results_*.json` — per-question predictions and references
- `batch_statistics_*.json` — F1, BLEU1, token usage, timing
- `summary.json` — overall metrics
