import json
import os
import re
import sys
import time
import datetime
import logging
import argparse
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locomo.retrievers import QdrantEntryLoader, VectorRetriever, format_related_memories
from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../data/ruler"))
DEFAULT_QDRANT_DIR = "./qdrant_post_update"
DEFAULT_RESULTS_DIR = "./results_ruler128k"
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"ruler128k_eval_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

QA_TASK_PREFIXES = ("qa_", "qa-")
RULER_TASK_FILES = [
    "cwe.jsonl", "fwe.jsonl",
    "niah_single_1.jsonl", "niah_single_2.jsonl", "niah_single_3.jsonl",
    "niah_multikey_1.jsonl", "niah_multikey_2.jsonl", "niah_multikey_3.jsonl",
    "niah_multivalue.jsonl", "niah_multiquery.jsonl",
    "vt.jsonl",
]

TASK_TO_METRIC = {
    "niah": "retri",
    "vt": "mt",
    "cwe": "agg",
    "fwe": "agg",
}

TINKER_API_KEY = os.getenv("OPENAI_API_KEY", "...")
TINKER_BASE_URL = os.getenv("OPENAI_BASE_URL", "...")
TINKER_LLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RUN_LOG_DIR, "ruler128k_evaluation.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ruler128k_eval")

def normalize_answer(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def is_correct(pred: str, gold_list: List[str]) -> bool:
    p = normalize_answer(pred)
    for g in gold_list:
        g = normalize_answer(str(g))
        if p == g or (g and g in p) or (p and p in g):
            return True
    if not gold_list:
        return False
    pred_tokens = set(p.split())
    for g in gold_list:
        gold_tokens = set(normalize_answer(str(g)).split())
        if gold_tokens and gold_tokens <= pred_tokens:
            return True
    return False

ANSWER_PROMPT = """Based on the following retrieved context, follow the instruction and answer. Reply with only the answer(s), no explanation.

Context:
{context}

Instruction / Question:
{question}

Answer:"""

def load_ruler_samples(data_root: str, task_files: list) -> List[tuple]:
    out = []
    for filename in task_files:
        path = os.path.join(data_root, filename)
        if not os.path.isfile(path):
            continue
        base = filename.replace(".jsonl", "").lower()
        task_type = "retri"
        for k, v in TASK_TO_METRIC.items():
            if base.startswith(k):
                task_type = v
                break
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = f"ruler128k-{base}-{idx}"
                out.append((raw, sample_id, task_type))
    return out

def main():
    parser = argparse.ArgumentParser(description="Evaluate LightMem on RULER 128k (Retri/MT/AGG Acc)")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--qdrant-dir", type=str, default=DEFAULT_QDRANT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval-limit", type=int, default=30)
    parser.add_argument("--metrics", type=str, default=None, help="Comma-separated metrics to run: retri,mt,agg (default: all)")
    parser.add_argument("--embedding-model-path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("OPENAI_API_KEY", TINKER_API_KEY))
    parser.add_argument("--llm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", TINKER_BASE_URL))
    parser.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL", TINKER_LLM_MODEL))
    parser.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"),
                        help="Device for embedding model (default: cuda)")
    args = parser.parse_args()

    dev = args.device
    if dev == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using cpu")
                dev = "cpu"
        except ImportError:
            dev = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    allowed_metrics = set(m.strip() for m in args.metrics.split(",")) if args.metrics else None
    task_files = [
        f for f in RULER_TASK_FILES
        if not any(f.lower().startswith(p) for p in QA_TASK_PREFIXES)
        and (allowed_metrics is None or TASK_TO_METRIC.get(
            next((k for k in TASK_TO_METRIC if f.lower().startswith(k)), ""), "retri"
        ) in allowed_metrics)
    ]
    samples = load_ruler_samples(args.data_root, task_files)
    if args.limit:
        samples = samples[: args.limit]
    logger.info("RULER 128k eval: %d samples, qdrant=%s, device=%s", len(samples), args.qdrant_dir, dev)

    embedder_cfg = BaseTextEmbedderConfig(
        model=args.embedding_model_path,
        embedding_dims=384,
        model_kwargs={"device": dev},
    )
    embedder = TextEmbedderHuggingface(embedder_cfg)
    entry_loader = QdrantEntryLoader(args.qdrant_dir, summary_suffix="_summary")
    retriever = VectorRetriever(embedder)
    llm = OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url) if args.llm_api_key else None

    correct = defaultdict(int)
    total = defaultdict(int)
    run_start = time.time()
    results_detail = []
    samples_with_entries = 0
    prompt_token_counts = []

    for raw, sample_id, task_type in tqdm(samples, desc="RULER 128k eval"):
        question = raw.get("question") or raw.get("instruction") or ""
        gold_list = raw.get("outputs", [])
        if not isinstance(gold_list, list):
            gold_list = [str(gold_list)]
        total[task_type] += 1

        pred = ""
        prompt_tokens = 0
        try:
            entries = entry_loader.load_entries(sample_id, with_vectors=True)
            if entries:
                samples_with_entries += 1
                retrieved = retriever.retrieve(entries, question, limit=args.retrieval_limit)
                context = format_related_memories(retrieved) if retrieved else ""
            else:
                context = ""
            prompt = ANSWER_PROMPT.format(context=context or "No context.", question=question)
            if llm:
                resp = llm.chat.completions.create(
                    model=args.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0,
                )
                pred = (resp.choices[0].message.content or "").strip()
                if resp.usage:
                    prompt_tokens = resp.usage.prompt_tokens
        except Exception as e:
            logger.warning("[%s] %s", sample_id, e)
        if prompt_tokens:
            prompt_token_counts.append(prompt_tokens)
        if is_correct(pred, gold_list):
            correct[task_type] += 1
        results_detail.append({"sample_id": sample_id, "task_type": task_type, "pred": pred, "gold": gold_list, "correct": is_correct(pred, gold_list), "prompt_tokens": prompt_tokens})

    running_time = time.time() - run_start
    end_ts = datetime.datetime.now().isoformat(timespec="seconds")

    logger.info("Samples with Qdrant entries: %d / %d (if this is small, run search with same --limit as add, or run add without --limit first)",
                samples_with_entries, len(samples))
    retri_acc = (correct["retri"] / total["retri"]) if total["retri"] else 0.0
    mt_acc = (correct["mt"] / total["mt"]) if total["mt"] else 0.0
    agg_acc = (correct["agg"] / total["agg"]) if total["agg"] else 0.0

    peak_prompt_tokens = max(prompt_token_counts) if prompt_token_counts else 0
    avg_prompt_tokens = int(sum(prompt_token_counts) / len(prompt_token_counts)) if prompt_token_counts else 0

    logger.info("=" * 60)
    logger.info("RULER(128k) — Retri. Acc.: %.4f (n=%d)", retri_acc, total["retri"])
    logger.info("RULER(128k) — MT Acc.:    %.4f (n=%d)", mt_acc, total["mt"])
    logger.info("RULER(128k) — AGG. Acc.:  %.4f (n=%d)", agg_acc, total["agg"])
    logger.info("Context window — peak: %d tokens, avg: %d tokens", peak_prompt_tokens, avg_prompt_tokens)
    logger.info("Running time: %.2f s", running_time)
    logger.info("Time (end): %s", end_ts)
    logger.info("=" * 60)

    stats = {
        "Retri_Acc": retri_acc,
        "MT_Acc": mt_acc,
        "AGG_Acc": agg_acc,
        "counts": {"retri": total["retri"], "mt": total["mt"], "agg": total["agg"]},
        "correct": dict(correct),
        "context_window_peak": {
            "solution_input_tokens_peak": peak_prompt_tokens,
            "solution_input_tokens_avg": avg_prompt_tokens,
        },
        "running_time_sec": round(running_time, 4),
        "time": end_ts,
        "log": {"results_dir": args.output_dir, "run_log_dir": RUN_TIMESTAMP},
    }
    stats_file = os.path.join(args.output_dir, "batch_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({**stats, "Retri_Acc": retri_acc, "MT_Acc": mt_acc, "AGG_Acc": agg_acc}, f, ensure_ascii=False, indent=2)
    detail_file = os.path.join(args.output_dir, "batch_results.json")
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(results_detail, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(args.output_dir, "experiment_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "outdir": os.path.abspath(args.output_dir),
                    "stats_file": os.path.abspath(stats_file),
                    "Retri_Acc": retri_acc,
                    "MT_Acc": mt_acc,
                    "AGG_Acc": agg_acc,
                    "solution_input_tokens_peak": peak_prompt_tokens,
                    "solution_input_tokens_avg": avg_prompt_tokens,
                    "running_time_sec": running_time,
                    "time": end_ts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    logger.info("Results: %s", args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
