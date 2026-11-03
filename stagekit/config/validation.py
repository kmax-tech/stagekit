# stagekit/config/validation.py

"""
Early configuration validation.

Two validators with distinct responsibilities:

    validate_app_config(cfg)
        Logical consistency — runs right after dictcfg_to_appcfg(),
        before any path resolution.  Catches structural mistakes that
        would otherwise surface deep inside the pipeline.

    validate_runtime_config(cfg)
        File-system reachability — runs after configure_paths(), when
        all paths are fully resolved.  Catches missing files/dirs that
        the pipeline would need at execution time.

Both collect errors into a list and raise a single ValueError at the
end, so the user sees every problem at once instead of fixing them
one at a time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from stagekit.utils.registry import LazyRegistry

if TYPE_CHECKING:
    from stagekit.config.classes import AppConfig

logger = logging.getLogger(__name__)

#: check name → module that registers it; each check is fn(pipeline) →
#: error messages and lives with the code whose assumption it holds. The
#: framework has none of its own; a topic adds its entries when loaded
#: (stagekit/run/topics.py).
PIPELINE_CHECK_MODULE_MAP: dict[str, str] = {}

PIPELINE_CHECK_REGISTRY = LazyRegistry("pipeline check", PIPELINE_CHECK_MODULE_MAP)


# ------------------------------------------------------------------
# Shared pipeline-structure helpers
# ------------------------------------------------------------------
# Used by both validate_app_config (early) and PipelineExecutor
# (defense-in-depth).  Logic lives here once; callers import it.
# ------------------------------------------------------------------

def check_unique_stage_names(stages) -> list[str]:
    """Return errors for duplicate stage names.

    *stages* is any iterable of objects with a ``.name`` attribute
    (PipelineStageConfig or equivalent).
    """
    errors: list[str] = []
    seen: dict[str, int] = {}
    for i, stage in enumerate(stages):
        if stage.name in seen:
            errors.append(
                f"Duplicate stage name '{stage.name}' "
                f"(positions {seen[stage.name]} and {i})"
            )
        seen[stage.name] = i
    return errors


def check_stage_dependencies(stages, preload_stages: set[str] | None = None) -> list[str]:
    """Return errors for depends_on references that point to unknown stages.

    *preload_stages* are stage names satisfied by preloaded artifacts —
    they don't appear in the pipeline config but their artifacts are
    already available at runtime.
    """
    known = {s.name for s in stages} | (preload_stages or set())
    errors: list[str] = []
    for stage in stages:
        for dep in stage.depends_on or []:
            if dep not in known:
                errors.append(
                    f"Stage '{stage.name}' depends on unknown stage '{dep}'. "
                    f"Available: {sorted(known)}"
                )
    return errors


def resolve_stage_type(stage) -> str:
    """Return the effective stage type for a PipelineStageConfig."""
    return stage.stage_type or stage.name


def get_known_stage_types() -> set[str]:
    """Return all registered + lazy-loadable stage type names."""
    from stagekit.pipeline.registry import STAGE_REGISTRY
    return set(STAGE_REGISTRY.available())


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def validate_app_config(cfg: AppConfig) -> None:
    """Logical consistency checks — no filesystem access."""
    errors: list[str] = []

    _check_fields_the_topic_sets(cfg, errors)
    _check_meta_config(cfg, errors)
    _check_data_config(cfg, errors)
    _check_model_config(cfg, errors)
    _check_llm_task_config(cfg, errors)
    _check_pipeline_config(cfg, errors)
    _check_stage_configs(cfg, errors)

    if errors:
        raise ConfigValidationError(errors)


def validate_runtime_config(cfg: AppConfig) -> None:
    """Post-path-resolution checks — filesystem reachability."""
    errors: list[str] = []

    _check_data_paths(cfg, errors)
    _check_stage_config_files(cfg, errors)
    _check_stage_paths(cfg, errors)

    if errors:
        raise ConfigValidationError(errors)


# ------------------------------------------------------------------
# Error type
# ------------------------------------------------------------------

class ConfigValidationError(ValueError):
    """Raised when configuration validation fails.

    Collects all errors into a single exception so the user can fix
    everything in one pass.
    """

    def __init__(self, errors: list[str]):
        bullet_list = "\n- ".join(errors)
        super().__init__(f"Invalid configuration:\n- {bullet_list}")
        self.errors = errors


# ------------------------------------------------------------------
# Logical checks (validate_app_config)
# ------------------------------------------------------------------

#: fields the framework leaves without a default; the topic's config.yaml
#: sets them (here: config/config.yaml)
FIELDS_THE_TOPIC_SETS = ("data.data_loader", "pipeline.variant")


def _check_fields_the_topic_sets(cfg: AppConfig, errors: list[str]) -> None:
    for dotted in FIELDS_THE_TOPIC_SETS:
        section, name = dotted.split(".")
        if not getattr(getattr(cfg, section), name):
            errors.append(
                f"{dotted} is not set — the framework has no default for it; "
                f"the topic's config.yaml sets it"
            )


def _check_meta_config(cfg: AppConfig, errors: list[str]) -> None:
    from stagekit.models.llm_cache import CACHE_MODES
    meta = cfg.meta
    if meta.cache_mode not in CACHE_MODES:
        errors.append(
            f"meta.cache_mode='{meta.cache_mode}' is not a cache mode "
            f"(available: {list(CACHE_MODES)})"
        )
    if meta.cache_mode == "replay" and not meta.enable_cache:
        errors.append(
            "meta.cache_mode='replay' with meta.enable_cache=false — a replay "
            "reads every answer from the cache"
        )
    if meta.replay_failed_items and meta.cache_mode != "replay":
        errors.append(
            f"meta.replay_failed_items is set but meta.cache_mode="
            f"'{meta.cache_mode}' — the list applies to a replay only"
        )


def _check_data_config(cfg: AppConfig, errors: list[str]) -> None:
    mode = cfg.data.mode

    if mode not in ("dataset", "file", "none"):
        errors.append(f"data.mode='{mode}' is invalid (expected 'dataset', 'file', or 'none')")
        return

    if mode == "file" and not cfg.data.file_path:
        errors.append("data.mode='file' requires data.file_path to be set")

    if mode == "file" and cfg.data.file_path:
        suffix = Path(cfg.data.file_path).suffix
        if suffix not in (".json", ".jsonl"):
            errors.append(
                f"data.file_path has unsupported extension '{suffix}' "
                f"(expected .json or .jsonl)"
            )

    if mode == "none" and cfg.pipeline.stages:
        # First stage must be able to produce data (canonical loaders, etc.)
        first = cfg.pipeline.stages[0]
        stage_type = first.stage_type or first.name
        # Canonical stages and certain derive stages produce data from disk,
        # so mode=none is valid when the first stage is one of those.
        # We check that requires_data is False on the first stage — if it
        # requires data, mode=none will blow up at runtime.
        _check_first_stage_produces_data(stage_type, errors)


def _check_first_stage_produces_data(stage_type: str, errors: list[str]) -> None:
    """Warn if mode=none but the first stage expects input data."""
    if stage_type not in get_known_stage_types():
        # Unknown stage — will be caught by _check_pipeline_config
        return

    from stagekit.pipeline.registry import get_stage_fn
    try:
        stage_cls = get_stage_fn(stage_type)
        if getattr(stage_cls, "requires_data", False):
            errors.append(
                f"data.mode='none' but the first pipeline stage '{stage_type}' "
                f"declares requires_data=True — no data will be available"
            )
    except ValueError:
        pass  # Unknown stage — caught elsewhere


def _check_model_config(cfg: AppConfig, errors: list[str]) -> None:
    model = cfg.model
    if model is None:
        return

    # If model fields are populated, check consistency
    if model.model_type and model.backend:
        from stagekit.models.registry import MODEL_REGISTRY
        if model.model_type not in MODEL_REGISTRY.available():
            errors.append(
                f"model.model_type='{model.model_type}' is not registered "
                f"(available: {MODEL_REGISTRY.available()})"
            )

    if model.backend and not model.model_name:
        errors.append(
            f"model.backend='{model.backend}' is set but model.model_name is empty"
        )

    if model.backend:
        # Dict membership, not BACKEND_REGISTRY.get(): the registry imports
        # the backend module, which a config check has no business doing.
        from stagekit.models.backends.factory import BACKEND_MODULE_MAP
        if model.backend not in BACKEND_MODULE_MAP:
            errors.append(
                f"model.backend='{model.backend}' is not a registered backend "
                f"(available: {sorted(BACKEND_MODULE_MAP)})"
            )

    if not model.supports_temperature and model.temperature != 1:
        errors.append(
            f"model '{model.model_name}': supports_temperature is false "
            f"but temperature={model.temperature} is set — remove the "
            f"override (the model rejects/ignores it; forcing it silently "
            f"would let the config lie about the run)"
        )


def _check_llm_task_config(cfg: AppConfig, errors: list[str]) -> None:
    llm_task = cfg.llm_task
    if llm_task is None:
        return

    if not llm_task.task_name:
        # LLM task with no task_name is a no-op default — skip
        return

    if not llm_task.template:
        errors.append(
            f"llm_task.task_name='{llm_task.task_name}' is set but "
            f"llm_task.template is missing"
        )

    if not llm_task.parser:
        errors.append(
            f"llm_task.task_name='{llm_task.task_name}' is set but "
            f"llm_task.parser is missing"
        )

    if llm_task.template:
        from stagekit.prompts.registry import PROMPT_REGISTRY
        if llm_task.template not in PROMPT_REGISTRY.available():
            errors.append(
                f"llm_task.template='{llm_task.template}' not found in prompt registry "
                f"(available: {PROMPT_REGISTRY.available()})"
            )

    if llm_task.parser:
        from stagekit.parsers.registry import PARSER_REGISTRY
        if llm_task.parser not in PARSER_REGISTRY.available():
            errors.append(
                f"llm_task.parser='{llm_task.parser}' not found in parser registry "
                f"(available: {PARSER_REGISTRY.available()})"
            )


def _check_pipeline_config(cfg: AppConfig, errors: list[str]) -> None:
    pipeline = cfg.pipeline

    if not pipeline.name and pipeline.stages:
        errors.append("pipeline.name must not be empty when stages are defined")

    if not pipeline.stages:
        return

    # Structural checks via shared helpers
    errors.extend(check_unique_stage_names(pipeline.stages))
    preload_names = set(pipeline.preload_stages.keys()) if pipeline.preload_stages else None
    errors.extend(check_stage_dependencies(pipeline.stages, preload_stages=preload_names))

    # Stage types must be resolvable
    known = get_known_stage_types()
    for stage in pipeline.stages:
        st = resolve_stage_type(stage)
        if st not in known:
            errors.append(
                f"stage '{stage.name}': stage_type='{st}' "
                f"is not registered in the stage registry"
            )

    # feature_sources / feature_views must be Dict[str, List[str]]
    for field_name, mapping in [
        ("feature_sources", pipeline.feature_sources),
        ("feature_views", pipeline.feature_views),
    ]:
        for key, value in mapping.items():
            if not isinstance(value, list) or not all(
                isinstance(v, str) for v in value
            ):
                errors.append(
                    f"pipeline.{field_name}['{key}'] must be a list of "
                    f"strings, got {type(value).__name__}"
                )

    # preload_artifacts: every value is pin:<coordinate>; the pin must
    # exist, its run must be on disk, and it must carry the artifact key.
    if pipeline.preload_artifacts:
        from stagekit.run import pins

        for artifact_key, spec in pipeline.preload_artifacts.items():
            errors.extend(pins.check_preload(spec, artifact_key))

    # checks the topic registers (variant values, multi-mode pins, ...), in
    # name order; each returns [] when its own condition does not apply
    # (e.g. source_mode_pins without source_modes)
    for name in PIPELINE_CHECK_REGISTRY.available():
        errors.extend(PIPELINE_CHECK_REGISTRY.get(name)(pipeline))

    # derived_artifacts: well-formed, known transform, source preloaded
    from stagekit.stages.derived_artifacts import check_derived_artifacts
    errors.extend(check_derived_artifacts(pipeline))


def _check_stage_configs(cfg: AppConfig, errors: list[str]) -> None:
    """Stage-specific business rules that can be checked without filesystem access.

    Delegated to each stage class's ``check_stage_config`` classmethod
    (see ``PipelineStage.check_stage_config``) so the rules live next to
    the stage that owns them, not in a stage-type ladder here.
    """
    from stagekit.pipeline.registry import get_stage_fn

    for stage in cfg.pipeline.stages:
        stage_type = resolve_stage_type(stage)
        try:
            stage_cls = get_stage_fn(stage_type)
        except (ValueError, ImportError):
            continue  # Unknown stage — already reported in _check_pipeline_config

        # Stages that require a model — check that one is resolvable
        _check_stage_model_resolvable(cfg, stage, stage_type, errors)

        errors.extend(stage_cls.check_stage_config(stage, cfg))


def _check_stage_model_resolvable(
    cfg: AppConfig,
    stage,
    stage_type: str,
    errors: list[str],
) -> None:
    """Check that stages declaring requires_model can resolve one.

    Validation rule:
    - Single-step LLM stages require a resolvable llm_task.
    - Workflow LLM stages (manages_own_prompts=True) require only a
      resolvable LLM model, because they provide parser/template
      explicitly per step.
    """
    from stagekit.pipeline.registry import get_stage_fn

    try:
        stage_cls = get_stage_fn(stage_type)
    except ValueError:
        return  # Unknown stage — already reported

    if not getattr(stage_cls, "requires_model", False):
        return

    # Stages that manage their own prompts (set parser/template internally)
    # need a model object but not an llm_task config.
    manages_own = getattr(stage_cls, "manages_own_prompts", False)

    # A model is resolvable if the stage or global config provides one
    has_stage_model = stage.model is not None
    has_global_model = (
        cfg.model is not None
        and cfg.model.model_type
        and cfg.model.backend
    )

    if not has_stage_model and not has_global_model:
        errors.append(
            f"stage '{stage.name}' (type={stage_type}) requires a model "
            f"but neither stage-level nor global model config is set"
        )

    # If the stage requires an LLM (has requires_model and is LLM-based),
    # check that an llm_task is resolvable
    has_stage_llm_task = stage.llm_task is not None
    has_global_llm_task = cfg.llm_task is not None and cfg.llm_task.task_name

    # Only warn about missing llm_task for LLM model types. For a
    # stage-level model override, load its YAML (a pure OmegaConf read —
    # no backend import) instead of skipping the sub-check: the skip let
    # a single-step LLM stage without any llm_task pass validation and
    # fail mid-run at template resolution (audit C5).
    if has_stage_model:
        from stagekit.config.io import load_model_config
        try:
            model_type = load_model_config(stage.model).model_type
        except Exception:  # noqa: BLE001 — any load failure, incl. OmegaConf's
            # Missing/broken model YAML is reported by
            # _check_stage_config_files; don't double-report here.
            return
    else:
        model_type = cfg.model.model_type if cfg.model else None

    if model_type == "llm" and not manages_own and not has_stage_llm_task and not has_global_llm_task:
        errors.append(
            f"stage '{stage.name}' (type={stage_type}) requires an LLM model "
            f"but no llm_task is configured (stage-level or global)"
        )


# ------------------------------------------------------------------
# Runtime checks (validate_runtime_config)
# ------------------------------------------------------------------

def _check_data_paths(cfg: AppConfig, errors: list[str]) -> None:
    mode = cfg.data.mode

    if mode == "file" and cfg.data.file_path:
        fp = Path(cfg.data.file_path)
        if not fp.exists():
            errors.append(f"data.file_path does not exist: {fp}")

    if mode == "dataset" and cfg.data.dataset_path:
        # dataset_path can be a logical name (handled by data_loader) or a
        # real path.  Only check if it looks like a filesystem path.
        dp = Path(cfg.data.dataset_path)
        if dp.suffix and not dp.exists():
            errors.append(f"data.dataset_path does not exist: {dp}")


def _check_stage_config_files(cfg: AppConfig, errors: list[str]) -> None:
    """Check that stage-level llm_task/model YAML files exist."""
    from stagekit.config.paths import CONFIG_DIR

    for stage in cfg.pipeline.stages:
        if stage.llm_task:
            llm_task_path = CONFIG_DIR / "llm_task" / f"{stage.llm_task}.yaml"
            if not llm_task_path.exists():
                errors.append(
                    f"stage '{stage.name}': llm_task='{stage.llm_task}' "
                    f"not found at {llm_task_path}"
                )

        if stage.model:
            model_path = CONFIG_DIR / "model" / f"{stage.model}.yaml"
            if not model_path.exists():
                errors.append(
                    f"stage '{stage.name}': model='{stage.model}' "
                    f"not found at {model_path}"
                )


def _check_stage_paths(cfg: AppConfig, errors: list[str]) -> None:
    """Check that stage config_override paths point to real files.

    WHICH keys are paths is declared on the stage class
    (``input_path_config_keys`` / ``output_path_config_keys``), not here —
    the central list this function used to carry drifted the moment a
    stage added a path key it didn't know about (``taxonomy_path``).
    """
    from stagekit.pipeline.registry import get_stage_fn

    for stage in cfg.pipeline.stages:
        stage_type = resolve_stage_type(stage)
        try:
            stage_cls = get_stage_fn(stage_type)
        except (ValueError, ImportError):
            continue  # Unknown stage — already reported in _check_pipeline_config

        co = stage.config_override or {}

        # input keys: must exist (file or directory)
        for key in sorted(stage_cls.input_path_keys()):
            value = co.get(key)
            if value and not Path(value).exists():
                errors.append(
                    f"stage '{stage.name}': {key} does not exist: {value}"
                )

        # output keys: the stage creates the file; parent dir must exist
        for key in sorted(stage_cls.output_path_keys()):
            value = co.get(key)
            if value and not Path(value).parent.exists():
                errors.append(
                    f"stage '{stage.name}': {key} parent directory "
                    f"does not exist: {Path(value).parent}"
                )
