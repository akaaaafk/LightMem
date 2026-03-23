import json
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

class BedrockChatCompletionMessage:
    def __init__(self, content: str):
        self.content = content

class BedrockChatCompletionChoice:
    def __init__(self, message: BedrockChatCompletionMessage):
        self.message = message

class BedrockUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

class BedrockChatCompletionResponse:

    def __init__(self, content: str, usage: Optional[BedrockUsage] = None):
        self.choices = [BedrockChatCompletionChoice(BedrockChatCompletionMessage(content))]
        self.usage = usage or BedrockUsage()

class BedrockOpenAIStyleClient:

    def __init__(
        self,
        region: str = "us-east-1",
        inference_profile_identifier: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        if boto3 is None:
            raise ImportError("boto3 required. pip install boto3")
        raw_id = inference_profile_identifier or model_id or "anthropic.claude-haiku-4-5-20251001-v1:0"
        self._region = region
        self._model_id = self._resolve_model_id(raw_id, region)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = boto3.client("bedrock-runtime", region_name=self._region)

    @staticmethod
    def _resolve_model_id(model_id: str, region: str) -> str:
        if model_id.startswith("arn:") or "/" in model_id or "." in model_id or ":" in model_id:
            return model_id
        try:
            import boto3 as _boto3
            account_id = _boto3.client("sts", region_name=region).get_caller_identity()["Account"]
            return f"arn:aws:bedrock:{region}:{account_id}:application-inference-profile/{model_id}"
        except Exception:
            return model_id

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> BedrockChatCompletionResponse:
        model_id = self._resolve_model_id(model, self._region) if model else self._model_id
        messages = messages or []
        max_tok = max_tokens if max_tokens is not None else self._max_tokens
        temp = temperature if temperature is not None else self._temperature

        converse_messages = []
        system_parts = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = (m.get("content") or "").strip()
            if role == "system":
                system_parts.append(content)
            else:
                converse_messages.append({
                    "role": "user" if role == "user" else "assistant",
                    "content": [{"text": content}],
                })
        if not converse_messages:
            return BedrockChatCompletionResponse("", BedrockUsage())

        params = {
            "modelId": model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tok,
                "temperature": temp,
            },
        }
        if system_parts:
            params["system"] = [{"text": "\n".join(system_parts)}]

        try:
            response = self._client.converse(**params)
        except ClientError as e:
            raise RuntimeError(f"Bedrock Converse error: {e}") from e

        output = response.get("output", {})
        message = output.get("message", {})
        content_list = message.get("content", [])
        text = ""
        if content_list:
            first = content_list[0]
            text = first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")

        usage = response.get("usage", {})
        prompt_tokens = usage.get("inputTokens", 0)
        completion_tokens = usage.get("outputTokens", 0)
        usage_obj = BedrockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return BedrockChatCompletionResponse(text, usage_obj)
