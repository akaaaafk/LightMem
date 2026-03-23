from openai import OpenAI
import json
import re
from tqdm import tqdm
import datetime
import time
import os
import logging
import argparse
import sqlite3
import shutil

from lightmem.memory.lightmem import LightMemory
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

DATA_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../data/hotpotqa/eval_1600.json"))
QDRANT_PRE_UPDATE_DIR = "./qdrant_pre_update"
QDRANT_POST_UPDATE_DIR = "./qdrant_post_update"
os.makedirs(QDRANT_PRE_UPDATE_DIR, exist_ok=True)
os.makedirs(QDRANT_POST_UPDATE_DIR, exist_ok=True)

API_KEYS = [
    os.getenv("OPENAI_API_KEY", "..."),
]
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "...")
LLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
LLMLINGUA_MODEL_PATH = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

def get_logger(sample_id=None):
    name = f"lightmem.hotpotqa.{sample_id or 'main'}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(RUN_LOG_DIR, f"{sample_id or 'main'}.log"), mode="w")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def split_context_to_messages(context: str, base_ts: str = "2024-01-01 00:00:00"):
    parts = re.split(r"(?=Document \d+:)", context, flags=re.IGNORECASE)
    messages = []
    base_dt = datetime.datetime.strptime(base_ts, "%Y-%m-%d %H:%M:%S")
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        ts = (base_dt + datetime.timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
        messages.append({"role": "user",      "content": part, "time_stamp": ts})
        messages.append({"role": "assistant", "content": "",   "time_stamp": ts})
    if not messages:
        messages = [
            {"role": "user",      "content": context[:50000], "time_stamp": base_ts},
            {"role": "assistant", "content": "",               "time_stamp": base_ts},
        ]
    return messages

def load_lightmem(collection_name: str, api_key: str,
                  base_dir: str = QDRANT_POST_UPDATE_DIR, device: str = "cpu",
                  metadata_generate: bool = True):
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": device,
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "instruction": "",
                    "rate": 0.6,
                    "target_token": -1,
                },
            },
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {"model_name": "llmlingua-2"},
        "metadata_generate": metadata_generate,
        "text_summary": False,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": LLM_MODEL,
                "api_key": api_key,
                "max_tokens": 4096,
                "openai_base_url": API_BASE_URL,
            },
        },
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": device},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f"{base_dir}/{collection_name}",
                "on_disk": True,
            },
        },
        "update": "offline",
        "extraction_mode": "flat",
    }
    return LightMemory.from_config(config)

def collection_entry_count(collection_name: str, base_dir: str) -> int:
    try:
        storage_sqlite = os.path.join(
            base_dir, collection_name, "collection", collection_name, "storage.sqlite"
        )
        if not os.path.exists(storage_sqlite):
            return 0
        conn = sqlite3.connect(storage_sqlite)
        cur = conn.execute("SELECT count(*) FROM points")
        row = cur.fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return -1

def process_single_sample(item: dict, index: int, api_key: str,
                           logger: logging.Logger, device: str = "cpu",
                           batch_docs: int = 20, metadata_generate: bool = True,
                           out_dir: str = QDRANT_POST_UPDATE_DIR):
    sample_id = f"hpqa-{index}"
    try:
        context = item.get("context", "")
        if not context:
            logger.warning(f"[{sample_id}] Empty context, skip")
            return {"sample_id": sample_id, "status": "skip", "error": "empty context"}

        messages = split_context_to_messages(context)
        n_docs = len(messages) // 2
        logger.info(f"[{sample_id}] {len(messages)} messages ({n_docs} docs), "
                    f"batch_docs={batch_docs} → ~{(n_docs + batch_docs - 1) // batch_docs} batches")

        lightmem = load_lightmem(sample_id, api_key, base_dir=out_dir, device=device,
                                 metadata_generate=metadata_generate)
        start = time.time()

        step = batch_docs * 2
        total = len(messages)
        for i in range(0, total, step):
            batch = messages[i: i + step]
            lightmem.add_memory(
                messages=batch,
                force_segment=True,
                force_extract=True,
            )
            logger.debug(f"[{sample_id}] Batch {i // step + 1}: {len(batch) // 2} docs processed")

        elapsed = time.time() - start
        count = collection_entry_count(sample_id, out_dir)
        logger.info(f"[{sample_id}] Done: {count} entries in {elapsed:.2f}s")

        return {
            "sample_id": sample_id,
            "status": "success",
            "entries": count,
            "total_duration": elapsed,
        }
    except Exception as e:
        logger.exception(f"[{sample_id}] Failed: {e}")
        return {"sample_id": sample_id, "status": "failed", "error": str(e)}

def parse_args():
    p = argparse.ArgumentParser(
        description="Build LightMem from HotpotQA eval_1600.json (full LightMem pipeline)"
    )
    p.add_argument("--data", type=str, default=DATA_PATH)
    p.add_argument("--limit", type=int, default=None,
                   help="Max samples (default: all; use 50 for quality eval)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"),
                   help="Device for LLMLingua + embedder (default: cpu)")
    p.add_argument("--batch-docs", type=int, default=20,
                   help="Documents per add_memory call (default: 20). "
                        "Larger = fewer API calls but bigger prompts. "
                        "Recommended range: 10–100.")
    p.add_argument("--no-extract", action="store_true",
                   help="Disable LLM extraction (metadata_generate=False). "
                        "Only LLMLingua + topic segmentation, no API calls.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory for Qdrant data "
                        "(default: qdrant_post_update, or noapi if --no-extract)")
    return p.parse_args()

def main():
    args = parse_args()
    logger = get_logger("main")

    metadata_generate = not args.no_extract
    out_dir = args.out_dir or ("./noapi" if args.no_extract else QDRANT_POST_UPDATE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output directory: %s", os.path.abspath(out_dir))
    n_batches_per_sample = (1600 + args.batch_docs - 1) // args.batch_docs
    n_samples = args.limit or 1600
    if metadata_generate:
        est_api_calls = n_samples * n_batches_per_sample
        est_hours = est_api_calls * 3 / 3600
        logger.info(
            "HotpotQA add_memory (LightMem pipeline, WITH extraction): "
            "batch_docs=%d, ~%d API calls/sample, "
            "~%d total calls, est %.1f hrs @ 3s/call",
            args.batch_docs, n_batches_per_sample, est_api_calls, est_hours,
        )
    else:
        logger.info(
            "HotpotQA add_memory (LightMem pipeline, NO extraction / no API): "
            "batch_docs=%d, ~%d batches/sample, est ~%.1f min/sample @ 3s/batch",
            args.batch_docs, n_batches_per_sample,
            n_batches_per_sample * 3 / 60,
        )

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = list(data.values()) if isinstance(data, dict) else []

    indices = range(args.start, min(args.start + (args.limit or len(data)), len(data)))
    samples = [(data[i], i) for i in indices]
    logger.info("Processing %d samples (start=%d)", len(samples), args.start)

    api_key = API_KEYS[0] or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No API key set.")

    start_time = time.time()
    results = []
    for item, idx in tqdm(samples, desc="HotpotQA add_memory"):
        res = process_single_sample(
            item, idx, api_key,
            get_logger(f"hpqa-{idx}"),
            device=args.device,
            batch_docs=args.batch_docs,
            metadata_generate=metadata_generate,
            out_dir=out_dir,
        )
        results.append(res)
    total_time = time.time() - start_time

    ok = [r for r in results if r.get("status") == "success"]
    logger.info("Done. Success=%d, Failed=%d, Wall time=%.2fs",
                len(ok), len(results) - len(ok), total_time)
    return 0 if results else 1

if __name__ == "__main__":
    main()
