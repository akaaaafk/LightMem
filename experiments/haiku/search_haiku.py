import json
import re
import os
import sys
from collections import Counter, defaultdict
from tqdm import tqdm
import datetime
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import argparse
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

from prompts import ANSWER_PROMPT, ANSWER_PROMPT_StructMem
from retrievers import QdrantEntryLoader, VectorRetriever, format_related_memories
from llm_judge import evaluate_llm_judge
from bedrock_client import BedrockOpenAIStyleClient

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"lightmem_haiku_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RUN_LOG_DIR, "haiku_evaluation.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("haiku_eval")

DEFAULT_DATA_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../data/locomo/locomo10.json"))
DEFAULT_QDRANT_DIR = "./qdrant_post_update"
DEFAULT_EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RESULTS_DIR = "./haiku_results"
DEFAULT_RETRIEVAL_LIMIT = 60
AWS_REGION = "us-east-1"
AWS_INFERENCE_PROFILE_ID = os.getenv("AWS_INFERENCE_PROFILE_ID", "...")

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

def bleu1_score_single(pred: str, gold: str) -> float:
    gtoks, ptoks = _tokens(gold), _tokens(pred)
    if not ptoks:
        return 0.0
    gcount, pcount = Counter(gtoks), Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    prec = clipped / len(ptoks)
    bp = 1.0 if len(ptoks) >= len(gtoks) else (math.exp(1 - len(gtoks) / len(ptoks)) if gtoks else 0.0)
    return bp * prec

def compute_metrics_by_category(items, pred_key="prediction", pred_field=None):
    agg = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        val = ex.get(pred_key, "")
        pred = val.get(pred_field, "") if pred_field and isinstance(val, dict) else (val if isinstance(val, str) else str(val or ""))
        f1, b1 = f1_score_single(pred, gold), bleu1_score_single(pred, gold)
        agg[cat].append((f1, b1))
        rows.append({"q_idx": idx, "category": cat, "gold_answer": str(gold), "prediction": str(pred), "F1": f1, "BLEU1": b1})
    summary = []
    for cat in sorted(agg.keys(), key=str):
        scores = agg[cat]
        if scores:
            summary.append({
                "category": cat,
                "count": len(scores),
                "F1_avg": sum(s[0] for s in scores) / len(scores),
                "BLEU1_avg": sum(s[1] for s in scores) / len(scores),
            })
    return summary, rows

def compute_f1(predictions: List[str], references: List[str]) -> float:
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    return float(np.mean([f1_score_single(p, r) for p, r in zip(predictions, references)]))

def compute_bleu1(predictions: List[str], references: List[str]) -> float:
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    return float(np.mean([bleu1_score_single(p, r) for p, r in zip(predictions, references)]))

def parse_locomo_dataset(data_path: str) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        sample = {"sample_id": item["sample_id"], "conversation": item["conversation"], "qa": []}
        for qa_item in item.get("qa", []):
            answer = qa_item.get("answer") or qa_item.get("adversarial_answer", "")
            sample["qa"].append({
                "question": qa_item["question"],
                "answer": answer,
                "category": qa_item["category"],
            })
        samples.append(sample)
    return samples

def retrieve_by_speaker(entries, retriever, question, limit_per_speaker):
    speaker_groups = {}
    for entry in entries:
        speaker_name = entry.get("payload", {}).get("speaker_name", "Unknown")
        speaker_groups.setdefault(speaker_name, []).append(entry)
    all_retrieved = []
    for speaker_name, group_entries in speaker_groups.items():
        speaker_retrieved = retriever.retrieve(group_entries, question, limit=limit_per_speaker)
        for entry in speaker_retrieved:
            entry["_retrieved_speaker"] = speaker_name
        all_retrieved.extend(speaker_retrieved)
    return all_retrieved

def retrieve_combined(entries, retriever, question, total_limit):
    combined = retriever.retrieve(entries, question, limit=total_limit)
    for entry in combined:
        entry["_retrieved_speaker"] = entry.get("payload", {}).get("speaker_name", "Unknown")
    return combined

def retrieve_summaries(summaries, retriever, question, limit):
    if not summaries:
        return []
    return retriever.retrieve(summaries, question, limit=limit)

def format_summaries(summaries: List[Dict]) -> str:
    if not summaries:
        return "No session summaries available."
    lines = []
    for summary in summaries:
        payload = summary.get("payload", {})
        lines.append(payload.get("summary", payload.get("memory", "")))
    return "\n".join(lines)

def build_prompt_with_speaker_memories(question, retrieved_entries, enable_summary=False, summaries=None):
    speaker_groups = {}
    for entry in retrieved_entries:
        speaker_name = entry.get("_retrieved_speaker", entry.get("payload", {}).get("speaker_name", "Unknown"))
        speaker_groups.setdefault(speaker_name, []).append(entry)
    speaker_names = list(speaker_groups.keys())
    if len(speaker_names) == 0:
        speaker_1_name, speaker_2_name = "Speaker 1", "Speaker 2"
        speaker_1_memories = speaker_2_memories = "No memories available."
    elif len(speaker_names) == 1:
        speaker_1_name, speaker_2_name = speaker_names[0], "Speaker 2"
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = "No memories available."
    else:
        speaker_1_name, speaker_2_name = speaker_names[0], speaker_names[1]
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = format_related_memories(speaker_groups[speaker_2_name])
    if enable_summary and summaries:
        session_summaries = format_summaries(summaries)
        return ANSWER_PROMPT_StructMem.format(
            speaker_1_name=speaker_1_name,
            speaker_1_memories=speaker_1_memories,
            speaker_2_name=speaker_2_name,
            speaker_2_memories=speaker_2_memories,
            session_summaries=session_summaries,
            question=question,
        )
    return ANSWER_PROMPT.format(
        speaker_1_name=speaker_1_name,
        speaker_1_memories=speaker_1_memories,
        speaker_2_name=speaker_2_name,
        speaker_2_memories=speaker_2_memories,
        question=question,
    )

def process_sample(
    sample,
    entry_loader,
    retriever,
    llm_client,
    judge_client,
    llm_model,
    judge_model,
    allow_categories,
    limit_per_speaker,
    total_limit,
    retrieval_mode,
    enable_summary=False,
    summary_limit=5,
):
    sample_id = sample["sample_id"]
    logger.info("Processing sample: %s", sample_id)
    sample_token_stats = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
    try:
        entries = entry_loader.load_entries(sample_id, with_vectors=True)
        summaries = entry_loader.load_summaries(sample_id, with_vectors=True) if enable_summary else []
    except Exception as e:
        logger.error("[%s] Failed to load: %s", sample_id, e)
        return {"sample_id": sample_id, "error": str(e), "results": [], "token_stats": sample_token_stats}
    if not entries:
        return {"sample_id": sample_id, "error": "No entries", "results": [], "token_stats": sample_token_stats}
    qa_results = []
    for qa in sample["qa"]:
        category = qa["category"]
        if int(category) == 5 or int(category) not in allow_categories:
            continue
        question = qa["question"]
        reference = qa["answer"]
        retrieved_summaries = retrieve_summaries(summaries, retriever, question, summary_limit) if (enable_summary and summaries) else []
        retrieved_entries = (
            retrieve_by_speaker(entries, retriever, question, limit_per_speaker)
            if retrieval_mode == "per-speaker"
            else retrieve_combined(entries, retriever, question, total_limit)
        )
        if not retrieved_entries:
            qa_results.append({
                "question": question,
                "prediction": "",
                "reference": reference,
                "category": category,
                "retrieved_count": 0,
                "summary_count": len(retrieved_summaries) if enable_summary else None,
                "retrieval_time": 0,
                "speaker_distribution": {},
                "error": "No entries retrieved",
                "metrics": {},
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
            continue
        speaker_dist = {}
        for entry in retrieved_entries:
            speaker = entry.get("_retrieved_speaker", "Unknown")
            speaker_dist[speaker] = speaker_dist.get(speaker, 0) + 1
        user_prompt = build_prompt_with_speaker_memories(
            question, retrieved_entries, enable_summary=enable_summary, summaries=retrieved_summaries if enable_summary else None
        )
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            generated_answer = response.choices[0].message.content
            for marker in ("# ANSWER:\n\n", "# ANSWER:\n", "# ANSWER:", "**Answer:**\n\n", "**Answer:**\n", "**Answer:**", "ANSWER:\n\n", "ANSWER:\n", "ANSWER:"):
                if marker in generated_answer:
                    generated_answer = generated_answer.split(marker, 1)[-1].strip()
                    break
            if getattr(response, "usage", None):
                token_usage["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0)
                token_usage["completion_tokens"] = getattr(response.usage, "completion_tokens", 0)
                token_usage["total_tokens"] = getattr(response.usage, "total_tokens", 0)
                sample_token_stats["total_prompt_tokens"] += token_usage["prompt_tokens"]
                sample_token_stats["total_completion_tokens"] += token_usage["completion_tokens"]
                sample_token_stats["total_tokens"] += token_usage["total_tokens"]
                sample_token_stats["api_calls"] += 1
        except Exception as e:
            logger.error("[%s] LLM failed: %s", sample_id, e)
            generated_answer = ""
        try:
            label = evaluate_llm_judge(question, reference, generated_answer, client_obj=judge_client, model_name=judge_model)
            metrics = {"judge_correct": int(label), "judge_response": "CORRECT" if int(label) == 1 else "WRONG"}
        except Exception as e:
            logger.error("[%s] Judge failed: %s", sample_id, e)
            metrics = {"judge_correct": 0, "judge_response": ""}
        qa_results.append({
            "question": question,
            "prediction": generated_answer,
            "reference": reference,
            "category": category,
            "retrieved_count": len(retrieved_entries),
            "speaker_distribution": speaker_dist,
            "retrieval_time": 0,
            "metrics": metrics,
            "token_usage": token_usage,
            **({"summary_count": len(retrieved_summaries)} if enable_summary else {}),
        })
    return {"sample_id": sample_id, "results": qa_results, "token_stats": sample_token_stats}

def main():
    parser = argparse.ArgumentParser(description="LoCoMo evaluation with AWS Bedrock (Haiku)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--qdrant-dir", type=str, default=DEFAULT_QDRANT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sample-size", type=int, default=None, help="Max number of samples (conversations) to evaluate")
    parser.add_argument("--limit-questions", type=int, default=None, help="Max number of questions (QA pairs) to evaluate; overrides sample-size when set")
    parser.add_argument("--limit-per-speaker", type=int, default=DEFAULT_RETRIEVAL_LIMIT)
    parser.add_argument("--total-limit", type=int, default=DEFAULT_RETRIEVAL_LIMIT)
    parser.add_argument("--retrieval-mode", type=str, choices=["combined", "per-speaker"], default="combined")
    parser.add_argument("--allow-categories", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--embedding-model-path", type=str, default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--embedding-device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--enable-summary", action="store_true")
    parser.add_argument("--summary-limit", type=int, default=5)
    parser.add_argument("--region", type=str, default=AWS_REGION)
    parser.add_argument("--inference-profile", type=str, default=AWS_INFERENCE_PROFILE_ID)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Haiku eval: dataset=%s qdrant=%s region=%s profile=%s", args.dataset, args.qdrant_dir, args.region, args.inference_profile)

    dev = "cuda" if args.embedding_device == "gpu" else args.embedding_device
    embedder_cfg = BaseTextEmbedderConfig(
        model=args.embedding_model_path,
        embedding_dims=384,
        model_kwargs={"device": dev},
    )
    embedder = TextEmbedderHuggingface(embedder_cfg)
    entry_loader = QdrantEntryLoader(args.qdrant_dir, summary_suffix="_summary") if args.enable_summary else QdrantEntryLoader(args.qdrant_dir)
    retriever = VectorRetriever(embedder)

    llm_client = BedrockOpenAIStyleClient(
        region=args.region,
        inference_profile_identifier=args.inference_profile,
        max_tokens=4096,
        temperature=0.0,
    )
    judge_client = BedrockOpenAIStyleClient(
        region=args.region,
        inference_profile_identifier=args.inference_profile,
        max_tokens=1024,
        temperature=0.0,
    )
    llm_model = judge_model = args.inference_profile

    samples = parse_locomo_dataset(args.dataset)
    if args.sample_size is not None:
        samples = samples[: args.sample_size]
    logger.info("Loaded %s samples", len(samples))

    eval_start = time.time()
    all_results = []
    all_predictions = []
    all_references = []
    all_categories = []
    all_questions = []
    total_questions = 0
    global_token_stats = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "total_api_calls": 0, "peak_prompt_tokens": 0, "peak_total_tokens": 0}

    for sample in tqdm(samples, desc="Haiku eval"):
        result = process_sample(
            sample,
            entry_loader,
            retriever,
            llm_client,
            judge_client,
            llm_model,
            judge_model,
            args.allow_categories,
            args.limit_per_speaker,
            args.total_limit,
            args.retrieval_mode,
            enable_summary=args.enable_summary,
            summary_limit=args.summary_limit,
        )
        all_results.append(result)
        tstats = result.get("token_stats", {})
        global_token_stats["total_prompt_tokens"] += tstats.get("total_prompt_tokens", 0)
        global_token_stats["total_completion_tokens"] += tstats.get("total_completion_tokens", 0)
        global_token_stats["total_tokens"] += tstats.get("total_tokens", 0)
        global_token_stats["total_api_calls"] += tstats.get("api_calls", 0)
        for qa_r in result.get("results", []):
            pt = qa_r.get("token_usage", {}).get("prompt_tokens", 0)
            tt = qa_r.get("token_usage", {}).get("total_tokens", 0)
            if pt > global_token_stats["peak_prompt_tokens"]:
                global_token_stats["peak_prompt_tokens"] = pt
            if tt > global_token_stats["peak_total_tokens"]:
                global_token_stats["peak_total_tokens"] = tt
        for qa_result in result.get("results", []):
            total_questions += 1
            all_predictions.append(qa_result.get("prediction", "") or "")
            all_references.append(qa_result.get("reference", "") or "")
            all_categories.append(qa_result["category"])
            all_questions.append(qa_result.get("question", "") or "")
            if args.limit_questions is not None and total_questions >= args.limit_questions:
                break
        if args.limit_questions is not None and total_questions >= args.limit_questions:
            all_predictions = all_predictions[: args.limit_questions]
            all_references = all_references[: args.limit_questions]
            all_categories = all_categories[: args.limit_questions]
            all_questions = all_questions[: args.limit_questions]
            total_questions = len(all_predictions)
        sample_file = os.path.join(args.output_dir, f"sample_{sample['sample_id']}.json")
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if args.limit_questions is not None and total_questions >= args.limit_questions:
            break

    running_time = time.time() - eval_start
    overall_f1 = compute_f1(all_predictions, all_references) if total_questions else 0.0
    overall_bleu1 = compute_bleu1(all_predictions, all_references) if total_questions else 0.0

    items = [
        {"question": all_questions[i], "gold_answer": all_references[i], "prediction": all_predictions[i], "category": all_categories[i]}
        for i in range(len(all_predictions))
    ]
    metrics_summary, metrics_rows = compute_metrics_by_category(items)
    batch_results_file = os.path.join(args.output_dir, "batch_results.json")
    with open(batch_results_file, "w", encoding="utf-8") as f:
        json.dump([{"question": all_questions[i], "gold_answer": all_references[i], "category": all_categories[i], "summary_answer": all_predictions[i]} for i in range(len(all_predictions))], f, ensure_ascii=False, indent=2)
    stats = {
        "total_samples": len(samples),
        "total_questions": total_questions,
        "overall_f1_avg": overall_f1,
        "overall_bleu1_avg": overall_bleu1,
        "by_category": metrics_summary,
        "input_tokens": global_token_stats["total_prompt_tokens"],
        "output_tokens": global_token_stats["total_completion_tokens"],
        "experiment_time_sec": round(running_time, 4),
    }
    stats_file = os.path.join(args.output_dir, "batch_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "llm": "AWS Bedrock (Haiku)",
            "inference_profile": args.inference_profile,
            "region": args.region,
            "total_questions": total_questions,
            "F1": overall_f1,
            "BLEU1": overall_bleu1,
            "running_time": running_time,
            "token_statistics": global_token_stats,
            **stats,
        }, f, ensure_ascii=False, indent=2)
    logger.info("F1=%.4f BLEU1=%.4f time=%.2fs results=%s", overall_f1, overall_bleu1, running_time, args.output_dir)

if __name__ == "__main__":
    main()
