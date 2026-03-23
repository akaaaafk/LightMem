import json
import re
import os
import sys
import time
import datetime
import logging
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locomo.retrievers import QdrantEntryLoader, VectorRetriever, format_related_memories
from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../data/hotpotqa/eval_1600.json"))
DEFAULT_QDRANT_DIR = "./qdrant_post_update"
DEFAULT_RESULTS_DIR = "./results_hotpotqa"
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"hotpotqa_eval_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RUN_LOG_DIR, "hotpotqa_evaluation.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("hotpotqa_eval")

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _tokens(s: str) -> List[str]:
    return normalize_text(s).split() if normalize_text(s) else []

def f1_score_single(pred: str, gold: str) -> float:
    gtoks, ptoks = _tokens(gold), _tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount, pcount = Counter(gtoks), Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    p, r = overlap / len(ptoks), overlap / len(gtoks)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def f1_score_multi_ref(pred: str, gold_list: List[str]) -> float:
    if not gold_list:
        return 0.0
    return max(f1_score_single(pred, g) for g in gold_list)

def average_f1(predictions: List[str], references: List[List[str]]) -> float:
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    return float(np.mean([f1_score_multi_ref(p, r) for p, r in zip(predictions, references)]))

ANSWER_PROMPT = """Based on the following retrieved context, answer the question concisely. Reply with only the answer, no explanation.

Context:
{context}

Question: {question}

Answer:"""

def main():
    parser = argparse.ArgumentParser(description="Evaluate LightMem on HotpotQA (F1, log, time)")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="eval_1600.json path")
    parser.add_argument("--qdrant-dir", type=str, default=DEFAULT_QDRANT_DIR, help="Qdrant post-update dir")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results output dir")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--retrieval-limit", type=int, default=30, help="Top-k retrieved memories")
    parser.add_argument("--embedding-model-path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"),
                        help="Device for embedding model (default: cuda)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("HotpotQA evaluation: data=%s, qdrant=%s", args.data, args.qdrant_dir)

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = list(data.values()) if isinstance(data, dict) else []
    if args.limit:
        data = data[: args.limit]
    logger.info("Samples to evaluate: %d", len(data))

    dev = args.device
    if dev == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                dev = "cpu"
        except ImportError:
            dev = "cpu"
    embedder_cfg = BaseTextEmbedderConfig(
        model=args.embedding_model_path,
        embedding_dims=384,
        model_kwargs={"device": dev},
    )
    embedder = TextEmbedderHuggingface(embedder_cfg)
    entry_loader = QdrantEntryLoader(args.qdrant_dir, summary_suffix="_summary")
    retriever = VectorRetriever(embedder)
    llm = OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url) if args.llm_api_key else None

    predictions: List[str] = []
    references: List[List[str]] = []
    run_start = time.time()
    peak_context_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    token_usage_per_sample: List[Dict[str, int]] = []

    for idx, item in enumerate(tqdm(data, desc="HotpotQA eval")):
        sample_id = f"hpqa-{idx}"
        question = item.get("input", "").strip()
        answers = item.get("answers", [])
        if not isinstance(answers, list):
            answers = [str(answers)]
        references.append(answers)

        pred = ""
        sample_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            entries = entry_loader.load_entries(sample_id, with_vectors=True)
            if not entries:
                logger.warning("[%s] No entries", sample_id)
            else:
                retrieved = retriever.retrieve(entries, question, limit=args.retrieval_limit)
                context = format_related_memories(retrieved) if retrieved else ""
                prompt = ANSWER_PROMPT.format(context=context or "No context.", question=question)
                if llm:
                    resp = llm.chat.completions.create(
                        model=args.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0,
                    )
                    pred = (resp.choices[0].message.content or "").strip()
                    if getattr(resp, "usage", None) is not None:
                        u = resp.usage
                        sample_usage = {
                            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                            "total_tokens": getattr(u, "total_tokens", 0) or 0,
                        }
                        peak_context_tokens = max(peak_context_tokens, sample_usage["prompt_tokens"])
                        total_prompt_tokens += sample_usage["prompt_tokens"]
                        total_completion_tokens += sample_usage["completion_tokens"]
        except Exception as e:
            logger.warning("[%s] %s", sample_id, e)
        predictions.append(pred)
        token_usage_per_sample.append(sample_usage)

    running_time = time.time() - run_start
    overall_f1 = average_f1(predictions, references)
    end_ts = datetime.datetime.now().isoformat(timespec="seconds")

    logger.info("=" * 60)
    logger.info("HotpotQA — average F1: %.4f", overall_f1)
    logger.info("Running time: %.2f s", running_time)
    logger.info("Context peak (prompt tokens): %d", peak_context_tokens)
    logger.info("Time (end): %s", end_ts)
    logger.info("=" * 60)

    batch_results = [
        {
            "index": i,
            "input": data[i].get("input"),
            "answers": references[i],
            "prediction": predictions[i],
            "token_usage": token_usage_per_sample[i] if i < len(token_usage_per_sample) else {},
        }
        for i in range(len(predictions))
    ]
    batch_file = os.path.join(args.output_dir, "batch_results.json")
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    stats = {
        "total_samples": len(data),
        "average_f1": overall_f1,
        "running_time_sec": round(running_time, 4),
        "time": end_ts,
        "context_peak_prompt_tokens": peak_context_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "log": {"results_dir": args.output_dir, "run_log_dir": RUN_TIMESTAMP},
    }
    stats_file = os.path.join(args.output_dir, "batch_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                **stats,
                "F1": overall_f1,
                "context_peak_windows": peak_context_tokens,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log_path = os.path.join(args.output_dir, "experiment_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "outdir": os.path.abspath(args.output_dir),
                    "batch_results_file": os.path.abspath(batch_file),
                    "stats_file": os.path.abspath(stats_file),
                    "total_samples": len(data),
                    "average_f1": overall_f1,
                    "running_time_sec": running_time,
                    "context_peak_prompt_tokens": peak_context_tokens,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
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
