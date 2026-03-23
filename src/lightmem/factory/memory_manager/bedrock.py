"""
AWS Bedrock memory manager. Uses Converse API with inference profile or model ID.
"""
import concurrent
import json
from typing import List, Dict, Optional, Literal, Any

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None

from lightmem.memory.prompts import EXTRACTION_PROMPTS, METADATA_GENERATE_PROMPT
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response


def _get_attr(cfg: Any, key: str, default: Any = None):
    return getattr(cfg, key, default) if cfg else default


def _resolve_model_id(model_id: str, region: str, account_id: Optional[str] = None) -> str:
    """
    If model_id looks like a short inference profile ID (no colon, no slash, no 'arn:'),
    construct the full application inference profile ARN.
    Falls back to using model_id as-is if account_id is not available.
    """
    if model_id.startswith("arn:"):
        return model_id  # already a full ARN
    if "/" in model_id or "." in model_id or ":" in model_id:
        return model_id  # looks like a full model ID or foundation model ARN
    # Short inference profile ID — resolve to full ARN if account_id known
    if account_id:
        return f"arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{model_id}"
    # Try to get account_id from STS
    try:
        import boto3 as _boto3
        sts = _boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]
        return f"arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{model_id}"
    except Exception:
        return model_id  # best effort


class BedrockManager:
    """LightMem memory manager using AWS Bedrock (Converse API)."""

    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config
        if boto3 is None:
            raise ImportError("boto3 is required for Bedrock. Install with: pip install boto3")

        raw_id = _get_attr(config, "inference_profile_identifier") or _get_attr(config, "model")
        if not raw_id:
            raw_id = "anthropic.claude-haiku-4-5-20251001-v1:0"
        self._region = _get_attr(config, "region", "us-east-1")
        # If short inference profile ID (no ARN prefix), resolve to full ARN
        self._model_id = _resolve_model_id(raw_id, self._region, _get_attr(config, "aws_account_id"))
        self._max_tokens = _get_attr(config, "max_tokens", 4096)
        self._temperature = _get_attr(config, "temperature", 0.1)
        self._top_p = _get_attr(config, "top_p", 0.9)
        self.context_windows = 200_000  # typical for Claude

        aws_kwargs = {"region_name": self._region}
        if _get_attr(config, "aws_access_key_id"):
            aws_kwargs["aws_access_key_id"] = config.aws_access_key_id
        if _get_attr(config, "aws_secret_access_key"):
            aws_kwargs["aws_secret_access_key"] = config.aws_secret_access_key

        try:
            self._client = boto3.client("bedrock-runtime", **aws_kwargs)
        except NoCredentialsError:
            raise ValueError(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
                "and AWS_REGION (or pass region in config)."
            )

    def _messages_to_converse(
        self, messages: List[Dict[str, str]]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Convert OpenAI-style messages to Bedrock Converse format. Returns (messages, system)."""
        converse_messages = []
        system_text = None
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = (m.get("content") or "").strip()
            if role == "system":
                system_text = (system_text or "") + content + "\n"
            elif role in ("user", "assistant"):
                converse_messages.append({
                    "role": "user" if role == "user" else "assistant",
                    "content": [{"text": content}],
                })
        if system_text:
            system_text = system_text.strip()
        return converse_messages, (system_text or None)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> tuple[Optional[str], Dict[str, int]]:
        """Call Bedrock Converse API. Returns (content, usage_info)."""
        converse_messages, system = self._messages_to_converse(messages)
        if not converse_messages:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        params: Dict[str, Any] = {
            "modelId": self._model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": self._max_tokens,
                "temperature": self._temperature,
                # topP omitted: Claude via Bedrock does not allow both temperature and topP
            },
        }
        if system:
            params["system"] = [{"text": system}]

        try:
            response = self._client.converse(**params)
        except ClientError as e:
            raise RuntimeError(f"Bedrock Converse error: {e}") from e

        if isinstance(response, dict):
            output = response.get("output", {})
            usage = response.get("usage", {})
        else:
            output = getattr(response, "output", None) or {}
            usage = getattr(response, "usage", None) or {}

        message = output.get("message", {}) if isinstance(output, dict) else getattr(output, "message", {})
        content_list = message.get("content", []) if isinstance(message, dict) else getattr(message, "content", [])
        text = ""
        if content_list:
            first = content_list[0]
            text = first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")

        if isinstance(usage, dict):
            prompt_tokens = usage.get("inputTokens", 0)
            completion_tokens = usage.get("outputTokens", 0)
        else:
            prompt_tokens = getattr(usage, "inputTokens", 0) or getattr(usage, "input_tokens", 0) or 0
            completion_tokens = getattr(usage, "outputTokens", 0) or getattr(usage, "output_tokens", 0) or 0
        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return text, usage_info

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only",
        topic_id_mapping: Optional[List[List[int]]] = None,
        extraction_mode: Literal["flat", "event"] = "flat",
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> List[Optional[Dict]]:
        """Extract metadata from segments (same contract as OpenaiManager)."""
        if not extract_list:
            return []
        default_prompts = EXTRACTION_PROMPTS.get(extraction_mode, {})
        prompts = {**default_prompts, **(custom_prompts or {})}
        if extraction_mode == "flat":
            return self._extract_with_prompt(
                system_prompt=prompts.get("factual", METADATA_GENERATE_PROMPT),
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual",
            )
        if extraction_mode == "event":
            factual_results = self._extract_with_prompt(
                system_prompt=prompts["factual"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual",
            )
            relational_results = self._extract_with_prompt(
                system_prompt=prompts["relational"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="relational",
            )
            return self._merge_dual_perspective_results(factual_results, relational_results)
        raise ValueError(f"Unknown extraction_mode: {extraction_mode}")

    def _merge_dual_perspective_results(
        self,
        factual_results: List[Optional[Dict]],
        relational_results: List[Optional[Dict]],
    ) -> List[Optional[Dict]]:
        merged = []
        for factual, relational in zip(factual_results, relational_results):
            if factual is None and relational is None:
                merged.append(None)
                continue
            m = {
                "input_prompt": [],
                "output_prompt": "",
                "cleaned_result": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            if factual:
                m["input_prompt"].extend(factual.get("input_prompt", []))
                m["cleaned_result"].extend(factual.get("cleaned_result", []))
                if factual.get("usage"):
                    for k in m["usage"]:
                        m["usage"][k] += factual["usage"].get(k, 0)
            if relational:
                m["input_prompt"].extend(relational.get("input_prompt", []))
                m["cleaned_result"].extend(relational.get("cleaned_result", []))
                if relational.get("usage"):
                    for k in m["usage"]:
                        m["usage"][k] += relational["usage"].get(k, 0)
            m["output_prompt"] = f"Factual: {factual.get('output_prompt', '') if factual else ''}\nRelational: {relational.get('output_prompt', '') if relational else ''}"
            merged.append(m)
        return merged

    def _extract_with_prompt(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: str,
        topic_id_mapping: Optional[List[List[int]]],
        entry_type: str = "factual",
    ) -> List[Optional[Dict]]:
        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            role_filter = {"user_only": {"user"}, "assistant_only": {"assistant"}, "hybrid": {"user", "assistant"}}
            allowed_roles = role_filter.get(messages_use, {"user"})
            lines = []
            for mes in segment:
                if mes.get("role") not in allowed_roles:
                    continue
                seq = mes.get("sequence_number", 0)
                role = mes.get("role", "")
                content = mes.get("content", "")
                speaker_name = mes.get("speaker_name", "")
                time_stamp = mes.get("time_stamp", "")
                weekday = mes.get("weekday", "")
                time_prefix = f"[{time_stamp}, {weekday}] " if (time_stamp and weekday) else ""
                if speaker_name:
                    lines.append(f"{time_prefix}{seq//2}.{speaker_name}: {content}")
                else:
                    lines.append(f"{time_prefix}{seq//2}.{role}: {content}")
            return "\n".join(lines)

        max_workers = min(len(extract_list), 5)

        def process_one(args):
            api_call_idx, api_call_segments = args
            try:
                user_prompt_parts = []
                global_topic_ids = topic_id_mapping[api_call_idx] if topic_id_mapping and api_call_idx < len(topic_id_mapping) else []
                for topic_idx, topic_segment in enumerate(api_call_segments):
                    gid = global_topic_ids[topic_idx] if topic_idx < len(global_topic_ids) else (topic_idx + 1)
                    topic_text = concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {gid} ---\n{topic_text}")
                user_prompt = "\n".join(user_prompt_parts)
                metadata_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                raw_response, usage_info = self.generate_response(
                    messages=metadata_messages,
                    response_format={"type": "json_object"},
                )
                metadata_facts = clean_response(raw_response)
                for entry in metadata_facts:
                    entry["entry_type"] = entry_type
                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": metadata_facts,
                    "usage": usage_info,
                    "entry_type": entry_type,
                }
            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}")
                return {"input_prompt": [], "output_prompt": "", "cleaned_result": [], "usage": None, "entry_type": entry_type}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_one, enumerate(extract_list)))
        return results

    def _call_update_llm(self, system_prompt: str, target_entry: Dict, candidate_sources: List[Dict]) -> Dict:
        target_memory = target_entry.get("payload", {}).get("memory", "")
        candidate_memories = [c.get("payload", {}).get("memory", "") for c in candidate_sources]
        user_prompt = f"Target memory:{target_memory}\nCandidate memories:\n" + "\n".join(f"- {m}" for m in candidate_memories)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response_text, usage_info = self.generate_response(messages=messages, response_format={"type": "json_object"})
            result = json.loads(response_text)
            if "action" not in result:
                result["action"] = "ignore"
            result["usage"] = usage_info
            return result
        except Exception:
            return {"action": "ignore", "usage": None}
