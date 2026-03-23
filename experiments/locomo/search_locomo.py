from openai import OpenAI
import json
import re
from collections import Counter, defaultdict
from tqdm import tqdm
import datetime
import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import argparse
import math

from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

from prompts import ANSWER_PROMPT, ANSWER_PROMPT_StructMem
from retrievers import QdrantEntryLoader, VectorRetriever, format_related_memories
from llm_judge import evaluate_llm_judge

LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"lightmem_locomo_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RUN_LOG_DIR, 'lightmem_locomo_evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vector_baseline")

DEFAULT_DATA_PATH = '/path/to/locomo_dataset.json'
DEFAULT_QDRANT_DIR = './qdrant_post_update' 
DEFAULT_EMBEDDING_MODEL_PATH = '/path/to/embedding-model'
DEFAULT_RESULTS_DIR = './lightmem_locomo_results'
DEFAULT_RETRIEVAL_LIMIT = 60

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
    bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks) / len(ptoks)) if gtoks else 0.0
    return bp * prec

def compute_metrics_by_category(
    items: List[Dict[str, Any]],
    pred_key: str = "prediction",
    pred_field: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    agg = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        val = ex.get(pred_key, "")
        if pred_field and isinstance(val, dict):
            pred = val.get(pred_field, "")
        else:
            pred = val if isinstance(val, str) else (str(val) if val is not None else "")
        f1, b1 = f1_score_single(pred, gold), bleu1_score_single(pred, gold)
        agg[cat].append((f1, b1))
        rows.append({
            "q_idx": idx,
            "category": cat,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1,
            "BLEU1": b1,
        })
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
    f1_scores = [f1_score_single(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(f1_scores))

def compute_bleu1(predictions: List[str], references: List[str]) -> float:
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    bleu_scores = [bleu1_score_single(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(bleu_scores))

def parse_locomo_dataset(data_path: str) -> List[Dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = {
            'sample_id': item['sample_id'],
            'conversation': item['conversation'],
            'qa': []
        }

        for qa_item in item.get('qa', []):
            answer = qa_item.get('answer') or qa_item.get('adversarial_answer', '')
            sample['qa'].append({
                'question': qa_item['question'],
                'answer': answer,
                'category': qa_item['category']
            })

        samples.append(sample)

    return samples

def retrieve_by_speaker(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    limit_per_speaker: int
) -> List[Dict]:
    speaker_groups = {}
    for entry in entries:
        payload = entry.get('payload', {})
        speaker_name = payload.get('speaker_name', 'Unknown')
        
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    logger.info(f"Found {len(speaker_groups)} speakers: {list(speaker_groups.keys())}")
    
    all_retrieved = []
    for speaker_name, group_entries in speaker_groups.items():
        logger.info(f"Retrieving top-{limit_per_speaker} for {speaker_name}...")
        
        speaker_retrieved = retriever.retrieve(
            group_entries, 
            question, 
            limit=limit_per_speaker
        )
        
        logger.info(f"  Retrieved {len(speaker_retrieved)}/{len(group_entries)} entries")
        
        for entry in speaker_retrieved:
            entry['_retrieved_speaker'] = speaker_name
        
        all_retrieved.extend(speaker_retrieved)
    
    return all_retrieved

def retrieve_combined(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    total_limit: int
) -> List[Dict]:
    logger.info(f"Retrieving combined top-{total_limit} entries across speakers...")
    
    combined = retriever.retrieve(entries, question, limit=total_limit)
    
    for entry in combined:
        payload = entry.get('payload', {})
        entry['_retrieved_speaker'] = payload.get('speaker_name', 'Unknown')
    
    logger.info(f"  Combined retrieval returned {len(combined)} entries")
    return combined

def retrieve_summaries(
    summaries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    limit: int
) -> List[Dict]:
    if not summaries:
        logger.debug("No summaries available")
        return []
    
    logger.debug(f"Retrieving top-{limit} from {len(summaries)} summaries")
    retrieved = retriever.retrieve(summaries, question, limit=limit)
    
    return retrieved

def format_summaries(summaries: List[Dict]) -> str:
    if not summaries:
        return "No session summaries available."
    
    lines = []
    for summary in summaries:
        payload = summary.get('payload', {})
        summary_text = payload.get('summary', payload.get('memory', ''))
        lines.append(f"{summary_text}")
    
    return "\n".join(lines)

def build_prompt_with_speaker_memories(
    question: str,
    retrieved_entries: List[Dict],
    enable_summary: bool = False,
    summaries: Optional[List[Dict]] = None
) -> str:
    speaker_groups = {}
    for entry in retrieved_entries:
        speaker_name = entry.get('_retrieved_speaker', 
                                 entry.get('payload', {}).get('speaker_name', 'Unknown'))
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    speaker_names = list(speaker_groups.keys())
    
    if len(speaker_names) == 0:
        speaker_1_name = "Speaker 1"
        speaker_2_name = "Speaker 2"
        speaker_1_memories = "No memories available."
        speaker_2_memories = "No memories available."
    elif len(speaker_names) == 1:
        speaker_1_name = speaker_names[0]
        speaker_2_name = "Speaker 2"
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = "No memories available."
    else:
        speaker_1_name = speaker_names[0]
        speaker_2_name = speaker_names[1]
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = format_related_memories(speaker_groups[speaker_2_name])
        
        logger.debug(
            f"Formatted memories - {speaker_1_name}: {len(speaker_groups[speaker_1_name])}, "
            f"{speaker_2_name}: {len(speaker_groups[speaker_2_name])}"
        )
    
    if enable_summary:
        session_summaries = format_summaries(summaries) if summaries else "No session summaries available."
        
        prompt = ANSWER_PROMPT_StructMem.format(
            speaker_1_name=speaker_1_name,
            speaker_1_memories=speaker_1_memories,
            speaker_2_name=speaker_2_name,
            speaker_2_memories=speaker_2_memories,
            session_summaries=session_summaries,
            question=question
        )
    else:
        prompt = ANSWER_PROMPT.format(
            speaker_1_name=speaker_1_name,
            speaker_1_memories=speaker_1_memories,
            speaker_2_name=speaker_2_name,
            speaker_2_memories=speaker_2_memories,
            question=question
        )
    
    return prompt

def process_sample(
    sample: Dict,
    entry_loader: QdrantEntryLoader,
    retriever: VectorRetriever,
    llm_client: OpenAI,
    judge_client: OpenAI,
    llm_model: str,
    judge_model: str,
    allow_categories: List[int],
    limit_per_speaker: int,
    total_limit: int,
    retrieval_mode: str,
    enable_summary: bool = False,
    summary_limit: int = 5
) -> Dict:
    sample_id = sample['sample_id']
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing sample: {sample_id}")
    logger.info(f"{'='*80}")
    
    sample_token_stats = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'api_calls': 0
    }
    
    try:
        entries = entry_loader.load_entries(sample_id, with_vectors=True)
        
        summaries = []
        if enable_summary:
            summaries = entry_loader.load_summaries(sample_id, with_vectors=True)
            logger.info(
                f"[{sample_id}] Loaded {len(entries)} entries + {len(summaries)} summaries"
            )
        else:
            logger.info(f"[{sample_id}] Loaded {len(entries)} entries")
        
        if not entries:
            logger.error(f"[{sample_id}] No entries loaded")
            return {
                'sample_id': sample_id,
                'error': 'No entries loaded',
                'results': [],
                'token_stats': sample_token_stats
            }
    except Exception as e:
        logger.error(f"[{sample_id}] Failed to load entries: {e}")
        return {
            'sample_id': sample_id,
            'error': str(e),
            'results': [],
            'token_stats': sample_token_stats
        }
    
    qa_results = []
    for qa_idx, qa in enumerate(sample['qa']):
        category = qa['category']
        
        if int(category) == 5 or category not in allow_categories:
            continue
        
        question = qa['question']
        reference = qa['answer']
        
        logger.info(f"\n[{sample_id}] Question {qa_idx+1} (Category {category})")
        logger.info(f"Q: {question}")
        logger.info(f"A: {reference}")
        
        time_start = time.time()
        
        retrieved_summaries = []
        if enable_summary and summaries:
            retrieved_summaries = retrieve_summaries(summaries, retriever, question, summary_limit)
        
        if retrieval_mode == 'per-speaker':
            retrieved_entries = retrieve_by_speaker(
                entries, retriever, question, limit_per_speaker
            )
        else:
            retrieved_entries = retrieve_combined(
                entries, retriever, question, total_limit
            )
        
        retrieval_time = time.time() - time_start
        
        if not retrieved_entries:
            logger.warning(f"[{sample_id}] No entries retrieved")
            qa_results.append({
                'question': question,
                'prediction': '',
                'reference': reference,
                'category': category,
                'retrieved_count': 0,
                'summary_count': 0 if enable_summary else None,
                'retrieval_time': retrieval_time,
                'speaker_distribution': {},
                'error': 'No entries retrieved',
                'metrics': {},
                'token_usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            })
            continue
        
        speaker_dist = {}
        for entry in retrieved_entries:
            speaker = entry.get('_retrieved_speaker', 'Unknown')
            speaker_dist[speaker] = speaker_dist.get(speaker, 0) + 1
        
        if enable_summary:
            logger.info(
                f"[{sample_id}] Retrieved {len(retrieved_summaries)} summaries + "
                f"{len(retrieved_entries)} entries in {retrieval_time:.3f}s"
            )
        else:
            logger.info(
                f"[{sample_id}] Retrieved {len(retrieved_entries)} entries in {retrieval_time:.3f}s"
            )
        logger.info(f"[{sample_id}] Speaker distribution: {speaker_dist}")
        
        user_prompt = build_prompt_with_speaker_memories(
            question, 
            retrieved_entries,
            enable_summary=enable_summary,
            summaries=retrieved_summaries if enable_summary else None
        )
        
        token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            generated_answer = response.choices[0].message.content
            
            if hasattr(response, 'usage') and response.usage:
                token_usage['prompt_tokens'] = response.usage.prompt_tokens
                token_usage['completion_tokens'] = response.usage.completion_tokens
                token_usage['total_tokens'] = response.usage.total_tokens
                
                sample_token_stats['total_prompt_tokens'] += token_usage['prompt_tokens']
                sample_token_stats['total_completion_tokens'] += token_usage['completion_tokens']
                sample_token_stats['total_tokens'] += token_usage['total_tokens']
                sample_token_stats['api_calls'] += 1
                
                logger.info(
                    f"[{sample_id}] Token usage - Prompt: {token_usage['prompt_tokens']}, "
                    f"Completion: {token_usage['completion_tokens']}, "
                    f"Total: {token_usage['total_tokens']}"
                )
            
            logger.info(f"[{sample_id}] Generated: {generated_answer}")
        except Exception as e:
            logger.error(f"[{sample_id}] Failed to generate answer: {e}")
            generated_answer = ""
        
        try:
            label = evaluate_llm_judge(
                question, reference, generated_answer,
                client_obj=judge_client, model_name=judge_model
            )
            metrics = {
                'judge_correct': int(label),
                'judge_response': 'CORRECT' if int(label) == 1 else 'WRONG'
            }
            logger.info(
                f"[{sample_id}] Judge: {'CORRECT' if int(label) == 1 else 'WRONG'}"
            )
        except Exception as e:
            logger.error(f"[{sample_id}] Judge evaluation failed: {e}")
            metrics = {'judge_correct': 0, 'judge_response': ''}
        
        result_dict = {
            'question': question,
            'prediction': generated_answer,
            'reference': reference,
            'category': category,
            'retrieved_count': len(retrieved_entries),
            'speaker_distribution': speaker_dist,
            'retrieval_time': retrieval_time,
            'metrics': metrics,
            'token_usage': token_usage
        }
        
        if enable_summary:
            result_dict['summary_count'] = len(retrieved_summaries)
        
        qa_results.append(result_dict)
    
    return {
        'sample_id': sample_id,
        'results': qa_results,
        'token_stats': sample_token_stats
    }

EXPERIMENT_LOG_FILENAME = "experiment_log.jsonl"

def log_experiment_run(
    outdir: str,
    batch_results_file: str,
    stats_file: str,
    argv: List[str],
    total_samples: int,
    total_questions: int,
    overall_f1: float,
    overall_bleu1: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    log_path = os.path.join(outdir, EXPERIMENT_LOG_FILENAME)
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "utc_ts": time.time(),
        "argv": argv,
        "outdir": os.path.abspath(outdir),
        "batch_results_file": os.path.abspath(batch_results_file),
        "stats_file": os.path.abspath(stats_file),
        "total_samples": total_samples,
        "total_questions": total_questions,
        "overall_f1_avg": overall_f1,
        "overall_bleu1_avg": overall_bleu1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"实验记录已追加: {log_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Vector baseline evaluation for LoCoMo dataset"
    )
    
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATA_PATH,
                       help="Path to dataset JSON file")
    parser.add_argument('--qdrant-dir', type=str, default=DEFAULT_QDRANT_DIR,
                       help="Path to Qdrant data directory")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULTS_DIR,
                       help="Output directory for results")
    parser.add_argument('--sample-size', type=int, default=None,
                       help="If set, only process first N samples (for quick testing)")
    
    parser.add_argument('--limit-per-speaker', type=int, default=DEFAULT_RETRIEVAL_LIMIT,
                       help="Retrieval limit per speaker (for per-speaker mode)")
    parser.add_argument('--total-limit', type=int, default=DEFAULT_RETRIEVAL_LIMIT,
                       help="Total retrieval limit (for combined mode)")
    parser.add_argument('--retrieval-mode', type=str, 
                       choices=['combined', 'per-speaker'], default='combined',
                       help="Retrieval strategy")
    parser.add_argument('--allow-categories', type=int, nargs='+', default=[1, 2, 3, 4],
                       help="Allowed QA categories")
    parser.add_argument('--embedder', type=str, 
                       choices=['huggingface', 'openai'], default='huggingface',
                       help="Embedding backend")
    parser.add_argument('--embedding-model-path', type=str, 
                       default=DEFAULT_EMBEDDING_MODEL_PATH,
                       help="Path to embedding model (for huggingface backend)")
    parser.add_argument('--embedding-device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'gpu'],
                       help="Device for embedding model (default: cpu)")
    
    parser.add_argument('--enable-summary', action='store_true',
                       help="Enable summary retrieval (StructMem mode)")
    parser.add_argument('--summary-limit', type=int, default=5,
                       help="Retrieval limit for summaries (only used if --enable-summary)")
    
    parser.add_argument('--llm-api-key', type=str, required=True,
                       help="API key for LLM")
    parser.add_argument('--llm-base-url', type=str, required=True,
                       help="Base URL for LLM API")
    parser.add_argument('--llm-model', type=str, required=True,
                       help="LLM model name")
    parser.add_argument('--judge-api-key', type=str, required=True,
                       help="API key for judge")
    parser.add_argument('--judge-base-url', type=str, required=True,
                       help="Base URL for judge API")
    parser.add_argument('--judge-model', type=str, required=True,
                       help="Judge model name")
    parser.add_argument('--prompt-price-per-1k', type=float, default=0.0,
                       help="Price per 1K prompt tokens (for cost estimation)")
    parser.add_argument('--completion-price-per-1k', type=float, default=0.0,
                       help="Price per 1K completion tokens (for cost estimation)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("LightMem Evaluation - LoCoMo Dataset")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset:         {args.dataset}")
    logger.info(f"  Qdrant dir:      {args.qdrant_dir}")
    logger.info(f"  Output dir:      {args.output_dir}")
    logger.info(f"  Retrieval mode:  {args.retrieval_mode}")
    if args.retrieval_mode == 'per-speaker':
        logger.info(f"  Limit per speaker: {args.limit_per_speaker}")
    else:
        logger.info(f"  Total limit:     {args.total_limit}")
    logger.info(f"  Summary enabled: {args.enable_summary}")
    if args.enable_summary:
        logger.info(f"  Summary limit:   {args.summary_limit}")
    logger.info(f"  Categories:      {args.allow_categories}")
    logger.info(f"  Embedder:        {args.embedder}")
    logger.info(f"  LLM model:       {args.llm_model}")
    logger.info(f"  Judge model:     {args.judge_model}")
    logger.info("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("\nInitializing components...")
    
    if args.enable_summary:
        entry_loader = QdrantEntryLoader(args.qdrant_dir, summary_suffix="_summary")
    else:
        entry_loader = QdrantEntryLoader(args.qdrant_dir)
    
    if args.embedder == 'openai':
        embedder_cfg = BaseTextEmbedderConfig(
            model='text-embedding-3-small',
            api_key=args.llm_api_key,
            openai_base_url=args.llm_base_url,
            embedding_dims=1536,
        )
        embedder = TextEmbedderOpenAI(embedder_cfg)
    else:
        dev = getattr(args, 'embedding_device', 'cpu')
        if dev == 'gpu':
            dev = 'cuda'
        embedder_cfg = BaseTextEmbedderConfig(
            model=args.embedding_model_path,
            embedding_dims=384,
            model_kwargs={"device": dev},
        )
        embedder = TextEmbedderHuggingface(embedder_cfg)
    
    retriever = VectorRetriever(embedder)
    
    llm_client = OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)
    judge_client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    
    logger.info(f"LLM client initialized: {args.llm_model}")
    logger.info(f"Judge client initialized: {args.judge_model}")
    
    logger.info(f"\nLoading dataset from {args.dataset}")
    samples = parse_locomo_dataset(args.dataset)
    logger.info(f"Loaded {len(samples)} samples")
    if getattr(args, 'sample_size', None) is not None:
        samples = samples[: args.sample_size]
        logger.info(f"Limited to first {len(samples)} samples (--sample-size={args.sample_size})")
    
    eval_start_time = time.time()
    
    global_token_stats = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'total_api_calls': 0
    }
    
    all_results = []
    all_metrics = []
    all_categories = []
    all_questions: List[str] = []
    all_predictions: List[str] = []
    all_references: List[str] = []
    total_questions = 0
    category_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_summaries_used = 0
    peak_context_tokens = 0
    
    for sample in tqdm(samples, desc="Processing samples"):
        sample_result = process_sample(
            sample, entry_loader, retriever,
            llm_client, judge_client,
            args.llm_model, args.judge_model,
            args.allow_categories, args.limit_per_speaker,
            args.total_limit, args.retrieval_mode,
            enable_summary=args.enable_summary,
            summary_limit=args.summary_limit
        )
        
        all_results.append(sample_result)
        
        sample_token_stats = sample_result.get('token_stats', {})
        global_token_stats['total_prompt_tokens'] += sample_token_stats.get('total_prompt_tokens', 0)
        global_token_stats['total_completion_tokens'] += sample_token_stats.get('total_completion_tokens', 0)
        global_token_stats['total_tokens'] += sample_token_stats.get('total_tokens', 0)
        global_token_stats['total_api_calls'] += sample_token_stats.get('api_calls', 0)
        
        for qa_result in sample_result.get('results', []):
            total_questions += 1
            category = qa_result['category']
            category_counts[category] += 1
            all_metrics.append(qa_result['metrics'])
            all_categories.append(category)
            all_questions.append(qa_result.get('question', "") or "")
            all_predictions.append(qa_result.get('prediction', "") or "")
            all_references.append(qa_result.get('reference', "") or "")
            token_usage = qa_result.get('token_usage', {}) or {}
            peak_context_tokens = max(
                peak_context_tokens,
                int(token_usage.get('total_tokens', 0) or 0),
            )
            
            if args.enable_summary and 'summary_count' in qa_result:
                total_summaries_used += qa_result['summary_count']
        
        sample_file = os.path.join(args.output_dir, f"sample_{sample['sample_id']}.json")
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_result, f, ensure_ascii=False, indent=2)
    
    logger.info("\nCalculating aggregate metrics...")
    category_scores = {}
    total_scores = []
    
    for cat, m in zip(all_categories, all_metrics):
        score = float(m.get('judge_correct', 0)) if isinstance(m, dict) else 0.0
        total_scores.append(score)
        category_scores.setdefault(int(cat), []).append(score)
    
    aggregate_results = {"overall": {}}
    if total_scores:
        aggregate_results["overall"]["judge_correct"] = {
            "mean": float(np.mean(total_scores)),
            "std": float(np.std(total_scores)),
            "count": int(len(total_scores)),
        }
    
    overall_f1 = compute_f1(all_predictions, all_references) if total_questions > 0 else 0.0
    overall_bleu1 = compute_bleu1(all_predictions, all_references) if total_questions > 0 else 0.0
    if total_scores:
        aggregate_results["overall"]["F1"] = overall_f1
        aggregate_results["overall"]["BLEU1"] = overall_bleu1
    
    for cat in sorted(category_scores.keys()):
        vals = category_scores[cat]
        if vals:
            aggregate_results[f"category_{cat}"] = {
                "judge_correct": {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "count": int(len(vals)),
                }
            }
    
    config_dict = {
        "retrieval_mode": args.retrieval_mode,
        "limit_per_speaker": args.limit_per_speaker,
        "total_limit": args.total_limit,
        "embedder": args.embedder,
        "method": "structmem" if args.enable_summary else "lightmem",
        "allow_categories": args.allow_categories,
        "enable_summary": args.enable_summary,
    }
    if args.enable_summary:
        config_dict["summary_limit"] = args.summary_limit
    
    eval_end_time = time.time()
    running_time = eval_end_time - eval_start_time
    end_timestamp_iso = datetime.datetime.now().isoformat(timespec="seconds")
    
    prompt_price = args.prompt_price_per_1k
    completion_price = args.completion_price_per_1k
    cost = 0.0
    if prompt_price or completion_price:
        cost = (
            (global_token_stats['total_prompt_tokens'] / 1000.0) * prompt_price +
            (global_token_stats['total_completion_tokens'] / 1000.0) * completion_price
        )
    
    n_samples = len(samples)
    start_idx, end_idx = 0, max(0, n_samples - 1)
    
    if total_questions > 0:
        items = [
            {
                "question": all_questions[i],
                "gold_answer": all_references[i],
                "prediction": all_predictions[i],
                "category": all_categories[i],
            }
            for i in range(len(all_predictions))
        ]
        metrics_summary, metrics_rows = compute_metrics_by_category(items)
        overall_f1 = sum(r["F1"] for r in metrics_rows) / len(metrics_rows)
        overall_bleu1 = sum(r["BLEU1"] for r in metrics_rows) / len(metrics_rows)
        
        batch_results_flat = [
            {
                "question": it["question"],
                "gold_answer": it["gold_answer"],
                "category": it["category"],
                "summary_answer": it["prediction"],
            }
            for it in items
        ]
        batch_results_file = os.path.join(args.output_dir, f"batch_results_{start_idx}_{end_idx}.json")
        with open(batch_results_file, "w", encoding="utf-8") as f:
            json.dump(batch_results_flat, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {batch_results_file}")

        tin, tout = global_token_stats["total_prompt_tokens"], global_token_stats["total_completion_tokens"]
        cost_per_sample = cost / n_samples if n_samples else 0.0
        token_per_sample_in = tin / n_samples if n_samples else 0.0
        token_per_sample_out = tout / n_samples if n_samples else 0.0
        token_per_sample_total = (tin + tout) / n_samples if n_samples else 0.0

        logger.info("\n" + "=" * 60)
        logger.info("Token 与费用")
        logger.info("=" * 60)
        logger.info(f"  input_tokens:  {tin}")
        logger.info(f"  output_tokens: {tout}")
        logger.info(f"  估算费用（花的钱）: ${cost:.4f}")
        logger.info("=" * 60)

        logger.info("\n" + "=" * 60)
        logger.info("Context Window Peak（单次调用最大 input_tokens）")
        logger.info("=" * 60)
        logger.info(f"  Memorize 阶段 peak:  0")
        logger.info(f"  Solution 阶段 peak: {peak_context_tokens}")
        logger.info("=" * 60)

        logger.info("\n" + "=" * 60)
        logger.info("Experiment Time / Cost & Token per Sample")
        logger.info("=" * 60)
        logger.info(f"  Running time:        {running_time:.2f} s")
        logger.info(f"  Cost per Sample:     ${cost_per_sample:.4f}")
        logger.info(f"  Token per Sample:    {token_per_sample_total:.0f} (input {token_per_sample_in:.0f} + output {token_per_sample_out:.0f})")
        logger.info("=" * 60)

        memorize_stats = {"phase": "memorize", "count": 0, "total_sec": 0.0, "avg_sec": 0.0}
        research_stats = {"phase": "research", "count": 0, "total_sec": 0.0, "avg_sec": 0.0}
        answer_stats = {"phase": "answer", "count": 1, "total_sec": round(running_time, 4), "avg_sec": round(running_time, 4)}
        logger.info("\n" + "=" * 60)
        logger.info("各阶段计时（每个 step 耗时 / 平均 / 总时间）")
        logger.info("=" * 60)
        for stat in (memorize_stats, research_stats, answer_stats):
            logger.info(f"  {stat['phase']}: count={stat['count']}, total={stat['total_sec']} s, avg={stat['avg_sec']} s")
        logger.info(f"  Total experiment: {running_time:.2f} s")
        logger.info("=" * 60)

        price_input_per_1m = prompt_price * 1000.0 if prompt_price else 0.0
        price_output_per_1m = completion_price * 1000.0 if completion_price else 0.0
        statistics = {
            "total_samples": n_samples,
            "total_questions": total_questions,
            "overall_f1_avg": overall_f1,
            "overall_bleu1_avg": overall_bleu1,
            "by_category": metrics_summary,
            "details": metrics_rows,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "input_tokens": tin,
            "output_tokens": tout,
            "cost_usd": round(cost, 6),
            "price_input_per_1m": round(price_input_per_1m, 6),
            "price_output_per_1m": round(price_output_per_1m, 6),
            "price_input_per_1k": prompt_price,
            "price_output_per_1k": completion_price,
            "context_window_peak": {
                "memorize_input_tokens": 0,
                "solution_input_tokens": peak_context_tokens,
            },
            "experiment_time_sec": round(running_time, 4),
            "running_time_sec": round(running_time, 4),
            "cost_per_sample": round(cost_per_sample, 6),
            "token_per_sample": {
                "input": round(token_per_sample_in, 2),
                "output": round(token_per_sample_out, 2),
                "total": round(token_per_sample_total, 2),
            },
            "timing": {
                "memorize": {"phase": "memorize", "count": 0, "total_sec": 0.0, "avg_sec": 0.0},
                "research": {"phase": "research", "count": 0, "total_sec": 0.0, "avg_sec": 0.0},
                "answer": {
                    "phase": "answer",
                    "count": 1,
                    "total_sec": round(running_time, 4),
                    "avg_sec": round(running_time, 4),
                },
                "total_experiment_sec": round(running_time, 4),
            },
        }
        stats_file = os.path.join(args.output_dir, f"batch_statistics_{start_idx}_{end_idx}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        logger.info(f"统计已保存: {stats_file}")
        
        log_experiment_run(
            outdir=args.output_dir,
            batch_results_file=batch_results_file,
            stats_file=stats_file,
            argv=sys.argv,
            total_samples=n_samples,
            total_questions=total_questions,
            overall_f1=overall_f1,
            overall_bleu1=overall_bleu1,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=cost,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("LoCoMo — 分数汇总（F1 / BLEU1）")
        logger.info("=" * 60)
        for r in metrics_summary:
            logger.info(f"  Category {r['category']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}")
        logger.info(f"\n整体: 问题数={total_questions}, 平均 F1={overall_f1:.4f}, 平均 BLEU1={overall_bleu1:.4f}")
        logger.info("=" * 60)
    
    final_results = {
        "llm_model": args.llm_model,
        "judge_model": args.judge_model,
        "dataset": args.dataset,
        "total_questions": total_questions,
        "total_samples": len(samples),
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "config": config_dict,
        "aggregate_metrics": aggregate_results,
        "running_time": running_time,
        "context_peak_windows": peak_context_tokens,
        "time": end_timestamp_iso,
        "F1": overall_f1,
        "BLEU1": overall_bleu1,
        "cost": cost,
        "tokens": {
            "prompt": global_token_stats['total_prompt_tokens'],
            "completion": global_token_stats['total_completion_tokens'],
            "total": global_token_stats['total_tokens'],
        },
        "log": {
            "results_dir": args.output_dir,
            "run_log_dir": RUN_TIMESTAMP,
        },
        "token_statistics": {
            "total_prompt_tokens": global_token_stats['total_prompt_tokens'],
            "total_completion_tokens": global_token_stats['total_completion_tokens'],
            "total_tokens": global_token_stats['total_tokens'],
            "total_api_calls": global_token_stats['total_api_calls'],
            "avg_prompt_tokens_per_call": (
                global_token_stats['total_prompt_tokens'] / global_token_stats['total_api_calls']
                if global_token_stats['total_api_calls'] > 0 else 0
            ),
            "avg_completion_tokens_per_call": (
                global_token_stats['total_completion_tokens'] / global_token_stats['total_api_calls']
                if global_token_stats['total_api_calls'] > 0 else 0
            ),
            "avg_total_tokens_per_call": (
                global_token_stats['total_tokens'] / global_token_stats['total_api_calls']
                if global_token_stats['total_api_calls'] > 0 else 0
            )
        },
        "timestamp": RUN_TIMESTAMP
    }
    
    if args.enable_summary:
        final_results["retrieval_statistics"] = {
            "total_summaries_used": total_summaries_used,
            "avg_summaries_per_question": total_summaries_used / total_questions if total_questions > 0 else 0,
        }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete")
    logger.info("=" * 80)
    logger.info(f"Total samples:    {len(samples)}")
    logger.info(f"Total questions:  {total_questions}")
    logger.info(f"LLM model:        {args.llm_model}")
    logger.info(f"Judge model:      {args.judge_model}")
    
    if args.enable_summary:
        logger.info("\nRetrieval Statistics:")
        logger.info(f"  Total summaries:  {total_summaries_used}")
        if total_questions > 0:
            logger.info(f"  Avg summaries/Q:  {total_summaries_used/total_questions:.2f}")
    
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        if count > 0:
            logger.info(
                f"  Category {category}: {count} questions "
                f"({count/total_questions*100:.1f}%)"
            )
    
    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict):
                logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"    {stat_name}: {value:.4f}")
    
    logger.info("\nToken Statistics:")
    logger.info(f"  Total API calls:        {global_token_stats['total_api_calls']}")
    logger.info(f"  Total prompt tokens:    {global_token_stats['total_prompt_tokens']:,}")
    logger.info(f"  Total completion tokens: {global_token_stats['total_completion_tokens']:,}")
    logger.info(f"  Total tokens:           {global_token_stats['total_tokens']:,}")
    if global_token_stats['total_api_calls'] > 0:
        logger.info(
            f"  Avg prompt/call:        "
            f"{global_token_stats['total_prompt_tokens'] / global_token_stats['total_api_calls']:.2f}"
        )
        logger.info(
            f"  Avg completion/call:    "
            f"{global_token_stats['total_completion_tokens'] / global_token_stats['total_api_calls']:.2f}"
        )
        logger.info(
            f"  Avg total/call:         "
            f"{global_token_stats['total_tokens'] / global_token_stats['total_api_calls']:.2f}"
        )
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()