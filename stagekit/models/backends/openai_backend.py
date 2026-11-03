# stagekit/models/backends/openai_backend.py
import logging

from openai import OpenAI

from stagekit.models.backends.backends_base import Answer, LLMBackend
from stagekit.models.backends.factory import register_backend

logger = logging.getLogger(__name__)


@register_backend("openai")
class OpenAIBackend(LLMBackend):
    name = "openai"

    #: Whether the model accepts temperature comes from the model YAML
    #: (ModelConfig.supports_temperature) — previously a hardcoded ID
    #: list here, which a new model snapshot silently escaped.
    config_fields = ("supports_temperature",)

    def __init__(
        self,
        model_name: str,
        temperature: float = 1,
        supports_temperature: bool = True,
    ):
        super().__init__(model_name, temperature)
        api_key = self.require_env("OPENAI_API_KEY", "OpenAI API key")

        if not supports_temperature and temperature != 1:
            # Config validation rejects this combination at preflight;
            # raising again here keeps the invariant when a backend is
            # built directly (tests, scripts).
            raise ValueError(
                f"Model '{model_name}' is declared supports_temperature: "
                f"false but temperature={temperature} was configured. "
                f"Remove the temperature override (the API would ignore "
                f"or reject it, and silently forcing it would let the "
                f"config lie about the run)."
            )

        self.client = OpenAI(api_key=api_key)
        self.supports_temperature = supports_temperature

        # Upper bound for concurrent requests; actual concurrency is
        # min(meta_cfg.max_workers, max_safe_workers).
        self.max_safe_workers = 300

    def generate(self, prompt: str) -> Answer:
        """Send one prompt and return the answer.

        Returns:
            The answer, with the reasoning trace when the server provides
            one: OpenAI-protocol reasoning endpoints (e.g. DeepSeek's
            ``deepseek-reasoner``, served over this backend) return it as
            ``message.reasoning_content``. Dropping it here would make
            this the one backend that loses the trace the Answer type
            exists to preserve; hosted OpenAI models simply lack the
            field, so ``thinking`` stays empty for them.
        """
        kwargs = self.chat_payload(prompt)

        if self.temperature is not None and self.temperature != 1:
            kwargs["temperature"] = self.temperature

        response = self.client.chat.completions.create(**kwargs)
        if not response.choices:
            raise ValueError("OpenAI returned empty choices array")
        message = response.choices[0].message
        return Answer(
            raw=message.content or "",
            thinking=getattr(message, "reasoning_content", None) or "",
        )
