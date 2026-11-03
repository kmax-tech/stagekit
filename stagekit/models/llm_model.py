# stagekit/models/llm_model.py

from pathlib import Path

from stagekit.models.base import LLMBaseModel
from stagekit.models.registry import register_model
from stagekit.parsers.registry import get_parser
from stagekit.prompts.registry import get_prompt
from stagekit.models.llm_cache import LLMCache


@register_model("llm")
class GenericLLMModel(LLMBaseModel):
    """
    Generic, task-driven LLM model.
    Fully defined by:
      - ModelConfig (backend + model_name + temperature)
      - LLMConfig (template, parser)
      - MetaConfig (cache, retries, workers)
    """

    def __init__(self, model_cfg, llm_task_cfg, meta_cfg):
        super().__init__(model_cfg, llm_task_cfg, meta_cfg)

        # -----------------------------
        # Resolve parser + template (only when llm_task is provided;
        # stages with manages_own_prompts set these themselves)
        # -----------------------------
        if llm_task_cfg is not None:
            self.parser = get_parser(llm_task_cfg.parser)
            self.template = get_prompt(llm_task_cfg.template)
        else:
            self.parser = None
            self.template = None

        # -----------------------------
        # Cache — independent of llm_task
        # -----------------------------
        if meta_cfg and meta_cfg.enable_cache:
            cache_dir = Path(meta_cfg.cache_dir)
            self.cache = LLMCache.get(cache_dir)
        else:
            self.cache = None
