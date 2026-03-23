import json
import os
import uuid
import time
import datetime
import logging
import argparse
from tqdm import tqdm

from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../data/ruler"))
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

QDRANT_POST_UPDATE_DIR = "./qdrant_post_update"
os.makedirs(QDRANT_POST_UPDATE_DIR, exist_ok=True)

QA_TASK_PREFIXES = ("qa_", "qa-")
RULER_TASK_FILES = [
    "cwe.jsonl",
    "fwe.jsonl",
    "niah_single_1.jsonl",
    "niah_single_2.jsonl",
    "niah_single_3.jsonl",
    "niah_multikey_1.jsonl",
    "niah_multikey_2.jsonl",
    "niah_multikey_3.jsonl",
    "niah_multivalue.jsonl",
    "niah_multiquery.jsonl",
    "vt.jsonl",
]

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000

def get_logger(name="main"):
    logger = logging.getLogger(f"ruler128k.{name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(RUN_LOG_DIR, f"{name}.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler())
    return logger

def context_to_chunks(context: str, chunk_size: int = CHUNK_SIZE) -> list:
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
    chunks = []
    start = 0
    i = 0
    while start < len(context):
        chunk = context[start: start + chunk_size].strip()
        start += chunk_size
        if not chunk:
            continue
        ts = (base_dt + datetime.timedelta(seconds=i)).isoformat(timespec="milliseconds")
        chunks.append({"content": chunk, "time_stamp": ts, "chunk_idx": i})
        i += 1
    return chunks

def load_qdrant(collection_name: str, base_dir: str = QDRANT_POST_UPDATE_DIR) -> Qdrant:
    cfg = QdrantConfig(
        collection_name=collection_name,
        embedding_model_dims=384,
        path=f"{base_dir}/{collection_name}",
        on_disk=True,
    )
    return Qdrant(cfg)

def load_ruler_samples(data_root: str, task_files: list) -> list:
    samples = []
    task_to_metric = {"niah": "retri", "vt": "mt", "cwe": "agg", "fwe": "agg"}
    for filename in task_files:
        path = os.path.join(data_root, filename)
        if not os.path.isfile(path):
            continue
        base = filename.replace(".jsonl", "").lower()
        task_type = "retri"
        for k, v in task_to_metric.items():
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
                samples.append((raw, sample_id, task_type))
    return samples

def process_one(raw: dict, sample_id: str, embedder: TextEmbedderHuggingface) -> dict:
    try:
        context = raw.get("example") or raw.get("context") or ""
        if not context:
            return {"sample_id": sample_id, "status": "skip", "error": "no context"}

        chunks = context_to_chunks(context)
        if not chunks:
            return {"sample_id": sample_id, "status": "skip", "error": "empty chunks"}

        qdrant = load_qdrant(sample_id)

        ids, vectors, payloads = [], [], []
        for chunk in chunks:
            vec = embedder.embed(chunk["content"])
            if vec is None:
                continue
            ids.append(str(uuid.uuid4()))
            vectors.append(vec)
            payloads.append({
                "memory": chunk["content"],
                "original_memory": chunk["content"],
                "compressed_memory": "",
                "time_stamp": chunk["time_stamp"],
                "float_time_stamp": datetime.datetime.fromisoformat(chunk["time_stamp"]).timestamp(),
                "weekday": "Mon",
                "topic_id": chunk["chunk_idx"],
                "topic_summary": "",
                "category": "",
                "subcategory": "",
                "memory_class": "",
                "speaker_id": "user",
                "speaker_name": "user",
                "hit_time": 0,
                "update_queue": [],
                "consolidated": False,
            })

        if not ids:
            return {"sample_id": sample_id, "status": "skip", "error": "all chunks failed embedding"}

        qdrant.insert(vectors=vectors, payloads=payloads, ids=ids)
        return {"sample_id": sample_id, "status": "success", "chunks": len(ids)}

    except Exception as e:
        import traceback
        return {"sample_id": sample_id, "status": "failed", "error": str(e), "traceback": traceback.format_exc()}

def parse_args():
    p = argparse.ArgumentParser(description="Build LightMem from RULER 128k (direct embed, no LLM)")
    p.add_argument("--data-root", type=str, default=DATA_ROOT)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tasks", type=str, nargs="*", default=None)
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"),
                   help="Device for embedding model (default: cuda; use cpu if no GPU or if cuda is slow)")
    return p.parse_args()

def main():
    args = parse_args()
    logger = get_logger("main")

    task_files = args.tasks or RULER_TASK_FILES
    task_files = [f for f in task_files if not any(f.lower().startswith(p) for p in QA_TASK_PREFIXES)]
    samples = load_ruler_samples(args.data_root, task_files)
    if args.limit:
        samples = samples[: args.limit]
    logger.info("RULER 128k add_memory: %d samples from %s", len(samples), args.data_root)

    dev = args.device
    if dev == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to cpu")
                dev = "cpu"
        except ImportError:
            dev = "cpu"
    logger.info("Loading embedder: %s (device=%s)", EMBEDDING_MODEL_PATH, dev)
    embedder_cfg = BaseTextEmbedderConfig(
        model=EMBEDDING_MODEL_PATH,
        embedding_dims=384,
        model_kwargs={"device": dev},
    )
    embedder = TextEmbedderHuggingface(embedder_cfg)
    logger.info("Embedder loaded.")

    start = time.time()
    results = []
    for raw, sample_id, task_type in tqdm(samples, desc="RULER add_memory"):
        res = process_one(raw, sample_id, embedder)
        results.append(res)

    elapsed = time.time() - start
    ok = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    total_chunks = sum(r.get("chunks", 0) for r in ok)
    logger.info("Done. Success=%d, Failed=%d, Total chunks=%d, Time=%.2fs",
                len(ok), len(failed), total_chunks, elapsed)
    if failed:
        logger.warning("First failed sample: %s — %s", failed[0]["sample_id"], failed[0].get("error", ""))
    return 0

if __name__ == "__main__":
    main()
