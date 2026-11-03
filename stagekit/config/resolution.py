# stagekit/config/resolution.py

import copy
from typing import List, cast

from stagekit.config.classes import (
    AppConfig,
    LLMConfig,
    PipelineConfig,
    PipelineStageConfig,
)
from stagekit.config.io import load_named_config, load_model_config, merge_to_dataclass
from stagekit.config.validation import validate_app_config


def dictcfg_to_appcfg(cfg_dictcfg) -> AppConfig:
    """Convert a Hydra/OmegaConf DictConfig into a typed AppConfig.

    Runs logical consistency validation (``validate_app_config``) before
    returning, so callers get early error messages for structural problems.
    """
    cfg_obj = cast(AppConfig, merge_to_dataclass(AppConfig, cfg_dictcfg))
    validate_app_config(cfg_obj)
    return cfg_obj


def resolve_pipeline_stages(
    pipeline_cfg: PipelineConfig,
) -> List[PipelineStageConfig]:
    """
    Takes a PipelineConfig with stages (list of PipelineStageConfig)
    and returns a NEW list of PipelineStageConfig with all
    model_cfg / llm_task_cfg resolved.
    """
    resolved = []

    for stage in pipeline_cfg.stages:
        # clone stage to avoid mutating the YAML-created object
        stg = PipelineStageConfig(
            name=stage.name,
            stage_type=stage.stage_type,
            model=stage.model,
            llm_task=stage.llm_task,
            config_override=copy.deepcopy(stage.config_override) if stage.config_override else None,
            depends_on=list(stage.depends_on) if stage.depends_on else None,
        )

        # Resolve model override (if any)
        if stg.model is not None:
            stg.model_cfg = load_model_config(stg.model)

        # Resolve LLM task override (if any)
        if stg.llm_task is not None:
            stg.llm_task_cfg = load_named_config(
                name=stg.llm_task,
                subdir="llm_task",
                cls=LLMConfig,
            )

        resolved.append(stg)

    return resolved
