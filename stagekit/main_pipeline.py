import logging
from pathlib import Path

from stagekit.config.classes import AppConfig
from stagekit.config.paths import load_project_env, project_path
from stagekit.pipeline.executor import PipelineExecutor
from stagekit.run.manager import RunManager
from stagekit.run.context import RunContext
from stagekit.run.utils import make_run_context, add_metadata, add_run_name
from stagekit.data_loader.registry import load_dataset
from stagekit.config.resolution import resolve_pipeline_stages
from stagekit.utils.serialization import save_json

logger = logging.getLogger(__name__)

# The project's .env (backend addresses, keys), in its root
# (stagekit/config/paths.py). Without a project there is none to load.
load_project_env()


def _setup_logging() -> None:
    """Configure default logging — safe to call multiple times (basicConfig is no-op after first call)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_run_context(cfg: AppConfig) -> RunContext:
    """Build the RunContext for this pipeline run (new unique timestamped directory)."""
    return make_run_context(cfg, run_type="pipeline", group=cfg.meta.run_group)


def _build_run_inputs(cfg: AppConfig) -> dict:
    """Extract human-readable run summary from resolved config."""
    pipeline = cfg.pipeline
    from stagekit.config.validation import resolve_stage_type
    stage_names = [
        f"{s.name} ({resolve_stage_type(s)})" if s.stage_type and s.stage_type != s.name
        else s.name
        for s in pipeline.stages
    ]

    inputs: dict = {
        "experiment": cfg.meta.experiment_name,
        "data": cfg.data.data_name,
        "data_mode": cfg.data.mode,
        "pipeline": pipeline.name,
    }

    # Config source paths (set by experiment runner)
    meta = cfg.meta
    if meta.experiment_config:
        inputs["experiment_config"] = meta.experiment_config
    if meta.data_config:
        inputs["data_config"] = meta.data_config
    if meta.pipeline_config:
        inputs["pipeline_config"] = meta.pipeline_config

    if pipeline.mode:
        inputs["mode"] = pipeline.mode
    if pipeline.variant:
        inputs["variant"] = pipeline.variant
    if pipeline.source_model:
        inputs["source_model"] = pipeline.source_model
    if pipeline.method:
        inputs["method"] = pipeline.method
    if pipeline.artifact_root:
        inputs["artifact_root"] = pipeline.artifact_root

    model = cfg.model
    if model and model.model_name:
        inputs["model"] = f"{model.backend}/{model.model_name}"
    if cfg.llm_task and cfg.llm_task.task_name:
        inputs["llm_task"] = cfg.llm_task.task_name

    if pipeline.preload_artifacts:
        inputs["preload_artifacts"] = dict(pipeline.preload_artifacts)
    if pipeline.preload_expect:
        inputs["preload_expect"] = dict(pipeline.preload_expect)
    if pipeline.derived_artifacts:
        inputs["derived_artifacts"] = {k: dict(v) for k, v in pipeline.derived_artifacts.items()}
    if pipeline.feature_views:
        inputs["feature_views"] = {k: list(v) for k, v in pipeline.feature_views.items()}
    if pipeline.feature_sources:
        inputs["feature_sources"] = {k: list(v) for k, v in pipeline.feature_sources.items()}

    inputs["stages"] = stage_names
    return inputs


def _log_and_save_run_inputs(cfg: AppConfig, run_dir: Path) -> None:
    """Log compact run summary and save run_inputs.yaml to run directory."""
    inputs = _build_run_inputs(cfg)

    # Compact log
    lines = [
        f"  Experiment:    {inputs['experiment']}",
        f"  Pipeline:      {inputs['pipeline']}",
        f"  Data:          {inputs['data']} ({inputs['data_mode']})",
    ]
    for key in ("experiment_config", "data_config", "pipeline_config"):
        if key in inputs:
            lines.append(f"  {key.replace('_', ' ').title():14s} {inputs[key]}")
    for key in ("mode", "variant", "source_model", "method", "model", "llm_task", "artifact_root"):
        if key in inputs:
            lines.append(f"  {key.replace('_', ' ').title():14s} {inputs[key]}")
    if "preload_artifacts" in inputs:
        lines.append("  Preloads:")
        for k, v in inputs["preload_artifacts"].items():
            lines.append(f"    {k} -> {v}")
    if "preload_expect" in inputs:
        lines.append("  Preload expect:")
        for k, v in inputs["preload_expect"].items():
            lines.append(f"    {k}: {v}")
    lines.append(f"  Stages:        {' -> '.join(inputs['stages'])}")

    logger.info("Run inputs:\n%s", "\n".join(lines))

    # Save to run dir
    from stagekit.utils.serialization import save_yaml
    save_yaml(inputs, run_dir / "run_inputs.yaml")


def run_pipeline(cfg: AppConfig):
    _setup_logging()

    # 1. Setup Context
    add_metadata(cfg)
    add_run_name(cfg)

    if not cfg.tracking.wandb_project:
        cfg.tracking.wandb_project = cfg.meta.experiment_name
    if not cfg.tracking.mlflow_experiment:
        cfg.tracking.mlflow_experiment = cfg.meta.experiment_name

    ctx = _resolve_run_context(cfg)
    ctx.run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("RUN: %s", ctx.run_name)
    logger.info("Output: %s", ctx.run_dir)

    _log_and_save_run_inputs(cfg, ctx.run_dir)

    # 2. Load Dataset
    #
    # Three data modes with different guarantees:
    #   "dataset" — through the configured data loader (the topic's; here
    #               the canonical JSONL loader, which checks component
    #               types): split-aware.  The standard
    #               path for classification experiments.
    #   "file"    — raw JSON/JSONL loaded verbatim, no schema validation,
    #               no splits.  For pipelines whose input records follow a
    #               different contract (e.g. tournament items).
    #   "none"    — empty list; pipelines that build data from scratch
    #               (canonical builder stages).
    data = []
    data_mode = cfg.data.mode

    if data_mode == "dataset":
        logger.info("Loading dataset: %s (%s)", cfg.data.dataset_path, cfg.data.split_id)
        dataset = load_dataset(cfg.data)
        data = dataset.to_records()
        logger.info("Loaded %d records", len(data))

    elif data_mode == "file":
        file_path = cfg.data.file_path
        if not file_path:
            raise ValueError("Data mode is 'file' but 'file_path' is not set.")
        else:
            logger.info("Loading file: %s", file_path)
            from stagekit.utils.serialization import load_jsonl, load_json

            fp = project_path(file_path)
            if not fp.exists():
                raise FileNotFoundError(f"Input file not found: {fp}")

            if fp.suffix == ".jsonl":
                data = load_jsonl(fp)
            elif fp.suffix == ".json":
                data = load_json(fp)
            else:
                raise ValueError(f"Unsupported file format: {fp.suffix}")

            logger.info("Loaded %d items from file", len(data))

    elif data_mode == "none":
        logger.info("Data mode is 'none' — starting with empty data")

    else:
        raise ValueError(f"Unknown data mode: '{data_mode}'. Expected 'dataset', 'file', or 'none'.")

    # 3. Resolve Configs
    cfg.pipeline.stages = resolve_pipeline_stages(cfg.pipeline)

    # 4. Initialize Experiment Manager
    run_manager = RunManager(cfg, ctx)

    # 5. Execute Pipeline
    executor = PipelineExecutor(
        cfg=cfg,
        run_manager=run_manager,
        data=data,
    )

    try:
        all_metrics = executor.run()
    except BaseException as exc:  # noqa: BLE001 — re-raised below
        # Failures are runs too: without a status and manifest the scanner
        # cannot see this run at all (include_failed would be dead code),
        # and open MLflow/W&B runs would leak. BaseException so that a
        # Ctrl-C'd run also lands in the ledger; always re-raised.
        _finalize_failed_run(run_manager, executor, exc)
        raise

    # 6. Save Summary
    save_json(
        {**all_metrics, **run_summary_fields(cfg, all_metrics)},
        run_manager.run_dir / "pipeline_summary.json",
    )
    if run_manager.tracker:
        run_manager.tracker.log_artifact(
            run_manager.run_dir / "pipeline_summary.json",
            artifact_path="summary",
        )

    logger.info("Pipeline complete -> %s", run_manager.run_dir)
    run_manager.finish()
    return run_manager.run_dir


def run_summary_fields(cfg: AppConfig, stage_metrics: dict) -> dict:
    """The run's own entries in pipeline_summary.json, beside each stage's metrics.

    ``llm_cache``: whether the run used stagekit's LLM cache
    (``meta.enable_cache``), how (``meta.cache_mode``) and, in a replay, the
    items it counted as failed instead of stopping
    (``meta.replay_failed_items``). A run on the labkit backend keeps its
    cache mode in the labkit run's record (``llm-cache.json``). ``labkit_run_id``: the labkit run this
    pipeline ran inside, None outside one.

    Raises:
        ValueError: A stage's name is one of these keys — its metrics would
            be overwritten.
    """
    fields = {
        "llm_cache": {
            "enabled": cfg.meta.enable_cache,
            "mode": cfg.meta.cache_mode,
            "replay_failed_items": list(cfg.meta.replay_failed_items),
        },
        "labkit_run_id": cfg.meta.labkit_run_id or None,
    }
    clash = sorted(set(fields) & set(stage_metrics))
    if clash:
        raise ValueError(
            f"pipeline_summary.json: stage name(s) {clash} collide with the "
            f"run's own entries — rename the stage(s)"
        )
    return fields


def _finalize_failed_run(
    run_manager: RunManager,
    executor: PipelineExecutor,
    exc: BaseException,
) -> None:
    """Record a crashed run in the ledger before the exception propagates.

    Never raises: recording the failure must not mask the failure itself.

    Side effects:
        Writes pipeline_status.json (status="failed" + error),
        run_manifest.json, and closes open tracker runs.
    """
    try:
        run_manager.write_pipeline_status(
            stage_names=getattr(executor, "completed_stages", []),
            status="failed",
            total_stages=len(executor.topological_order),
            error=f"{type(exc).__name__}: {exc}",
        )
        run_manager.finish()
        logger.error("Pipeline FAILED -> %s", run_manager.run_dir)
    except Exception:
        logger.exception("Could not record the failed run in the ledger")
