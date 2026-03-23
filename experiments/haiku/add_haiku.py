import json
from tqdm import tqdm
import datetime
import time
import os
import logging
from lightmem.memory.lightmem import LightMemory
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from prompts import METADATA_GENERATE_PROMPT_locomo, LoCoMo_Event_Binding_factual, LoCoMo_Event_Binding_relational
import sqlite3
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

AWS_REGION = "us-east-1"
AWS_INFERENCE_PROFILE_ID = os.getenv("AWS_INFERENCE_PROFILE_ID", "...")

LLMLINGUA_MODEL_PATH = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../data/locomo/locomo10.json"))

QDRANT_PRE_UPDATE_DIR = "./qdrant_pre_update"
QDRANT_POST_UPDATE_DIR = "./qdrant_post_update"
os.makedirs(QDRANT_PRE_UPDATE_DIR, exist_ok=True)
os.makedirs(QDRANT_POST_UPDATE_DIR, exist_ok=True)

MAX_WORKERS = 5
USE_PROCESS_POOL = True

def parse_args():
    parser = argparse.ArgumentParser(description="LightMem + LoCoMo with AWS Bedrock (Haiku)")
    parser.add_argument("--extraction_mode", type=str, default="flat", choices=["flat", "event"])
    parser.add_argument("--enable_summary", action="store_true")
    parser.add_argument("--summary_time_window", type=int, default=3600)
    parser.add_argument("--summary_top_k_seeds", type=int, default=15)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--region", type=str, default=AWS_REGION, help="AWS region for Bedrock")
    parser.add_argument("--inference-profile", type=str, default=AWS_INFERENCE_PROFILE_ID,
                        help="Bedrock inference profile identifier")
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples to process (default: all)")
    return parser.parse_args()

def get_process_logger(sample_id):
    logger = logging.getLogger(f"lightmem.haiku.{sample_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(RUN_LOG_DIR, f"{sample_id}.log"), mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def parse_locomo_timestamp(timestamp_str):
    timestamp_str = timestamp_str.strip("()")
    try:
        dt = datetime.datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str

def extract_locomo_sessions(conversation_dict):
    speaker_a = conversation_dict.get("speaker_a", "Speaker_A")
    speaker_b = conversation_dict.get("speaker_b", "Speaker_B")
    session_nums = set()
    for key in conversation_dict.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.split("_")[1])
                session_nums.add(num)
            except Exception:
                continue
    sessions = []
    timestamps = []
    for num in sorted(session_nums):
        session_key = f"session_{num}"
        timestamp_key = f"{session_key}_date_time"
        if session_key not in conversation_dict:
            continue
        session_data = conversation_dict[session_key]
        timestamp = conversation_dict.get(timestamp_key, "")
        messages = []
        for turn in session_data:
            speaker_name = turn["speaker"]
            speaker_id = "speaker_a" if speaker_name == speaker_a else "speaker_b"
            content = turn["text"]
            if turn.get("blip_caption"):
                content = f"{content} (image description: {turn['blip_caption']})"
            messages.append({"role": "user", "content": content, "speaker_id": speaker_id, "speaker_name": speaker_name})
            messages.append({"role": "assistant", "content": "", "speaker_id": speaker_id, "speaker_name": speaker_name})
        sessions.append(messages)
        timestamps.append(parse_locomo_timestamp(timestamp))
    return sessions, timestamps, speaker_a, speaker_b

def load_lightmem(collection_name, args, base_dir=QDRANT_POST_UPDATE_DIR):
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cpu",
                    "use_llmlingua2": True,
                },
                "compress_config": {"instruction": "", "rate": 0.6, "target_token": -1},
            },
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {"model_name": "llmlingua-2"},
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "bedrock",
                "configs": {
                    "region": args.region,
                    "inference_profile_identifier": args.inference_profile,
                    "max_tokens": 16000,
                    "temperature": 0.1,
                },
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cpu"},
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
        "summary_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": f"{collection_name}_summary",
                "embedding_model_dims": 384,
                "path": f"{base_dir}/{collection_name}_summary",
                "on_disk": True,
            },
        },
        "update": "offline",
        "logging": {"level": "DEBUG", "file_enabled": True, "log_dir": RUN_LOG_DIR},
        "extraction_mode": args.extraction_mode,
    }
    return LightMemory.from_config(config)

def collection_entry_count(collection_name, base_dir):
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

def process_single_sample(sample, args):
    sample_id = sample["sample_id"]
    logger = get_process_logger(sample_id)
    prompt_arg = (
        {"factual": LoCoMo_Event_Binding_factual, "relational": LoCoMo_Event_Binding_relational}
        if args.extraction_mode == "event"
        else METADATA_GENERATE_PROMPT_locomo
    )
    try:
        logger.info("Processing: %s (Bedrock Haiku)", sample_id)
        conversation = sample["conversation"]
        sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(conversation)
        lightmem = load_lightmem(collection_name=sample_id, args=args)
        case_start = time.time()
        for session, timestamp in zip(sessions, timestamps):
            while session and session[0].get("role") != "user":
                session.pop(0)
            num_turns = len(session) // 2
            for turn_idx in range(num_turns):
                turn_messages = session[turn_idx * 2 : turn_idx * 2 + 2]
                if len(turn_messages) < 2 or turn_messages[0].get("role") != "user":
                    continue
                for msg in turn_messages:
                    msg["time_stamp"] = timestamp
                is_last = session is sessions[-1] and turn_idx == num_turns - 1
                lightmem.add_memory(
                    messages=turn_messages,
                    METADATA_GENERATE_PROMPT=prompt_arg,
                    force_segment=is_last,
                    force_extract=is_last,
                )
        add_duration = time.time() - case_start
        after_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
        logger.info("Add_memory completed: %s entries in %.2fs", after_count, add_duration)
        if os.path.exists(f"{QDRANT_PRE_UPDATE_DIR}/{sample_id}"):
            shutil.rmtree(f"{QDRANT_PRE_UPDATE_DIR}/{sample_id}")
        shutil.copytree(
            f"{QDRANT_POST_UPDATE_DIR}/{sample_id}",
            f"{QDRANT_PRE_UPDATE_DIR}/{sample_id}",
            ignore=shutil.ignore_patterns("*.lock"),
        )
        if args.enable_summary:
            lightmem_summary = load_lightmem(sample_id, args, base_dir=QDRANT_PRE_UPDATE_DIR)
            lightmem_summary.summarize(
                retrieval_scope="global",
                time_window=args.summary_time_window,
                top_k_seeds=args.summary_top_k_seeds,
                process_all=True,
            )
        lightmem.construct_update_queue_all_entries()
        lightmem.offline_update_all_entries(score_threshold=0.9)
        post_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
        logger.info("Done: %s pre=%s post=%s", sample_id, after_count, post_count)
        return {
            "sample_id": sample_id,
            "status": "success",
            "pre_update_count": after_count,
            "post_update_count": post_count,
            "total_duration": time.time() - case_start,
        }
    except Exception as e:
        logger.exception("Failed: %s", e)
        return {"sample_id": sample_id, "status": "failed", "error": str(e)}

def main():
    args = parse_args()
    main_logger = logging.getLogger("lightmem.haiku.main")
    main_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(RUN_LOG_DIR, "main.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    main_logger.addHandler(fh)
    main_logger.addHandler(logging.StreamHandler())
    main_logger.info("Haiku (AWS Bedrock) region=%s inference_profile=%s", args.region, args.inference_profile)
    data = json.load(open(DATA_PATH, "r"))
    missing = []
    for sample in data:
        sample_id = sample["sample_id"]
        post_dir = f"{QDRANT_POST_UPDATE_DIR}/{sample_id}"
        if os.path.exists(post_dir) and collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR) > 0:
            continue
        missing.append(sample)
    if args.limit is not None:
        missing = missing[: args.limit]
        main_logger.info("Limited to first %s samples", args.limit)
    main_logger.info("Samples to process: %s", len(missing))
    if not missing:
        main_logger.info("All complete.")
        return
    start_time = time.time()
    results = []
    ExecutorClass = ProcessPoolExecutor if USE_PROCESS_POOL else ThreadPoolExecutor
    with ExecutorClass(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_sample, s, args): s for s in missing}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                main_logger.exception("%s", e)
    ok = [r for r in results if r.get("status") == "success"]
    main_logger.info("Done. Success=%s Failed=%s Wall=%.2fs", len(ok), len(results) - len(ok), time.time() - start_time)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
