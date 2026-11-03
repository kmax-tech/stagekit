"""Preflight validation and DAG visualization for pipelines.

Runs everything the executor validates at DAG build time — stage name /
dependency checks, topological sort (cycle detection), enrichment and
artifact contracts, preload-stage resolution, and preloaded-artifact
integrity (existence, hashes, schema) — WITHOUT executing a single
stage, loading a model, or writing to ``output/``.

Used by ``run_experiment --dry-run`` to inspect an experiment's combos
before committing compute, and to render the pipeline DAG (ASCII for
the terminal, Mermaid for pasting into Markdown).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from stagekit.config.classes import AppConfig, PipelineStageConfig
from stagekit.pipeline.executor import PipelineExecutor
from stagekit.pipeline.registry import get_stage_fn
from stagekit.stages.artifact_schemas import ArtifactSchemaError
from stagekit.config.resolution import resolve_pipeline_stages
from stagekit.config.validation import resolve_stage_type

logger = logging.getLogger(__name__)


class _NullRunManager:
    """Stand-in for RunManager during preflight.

    ``PipelineExecutor.__init__`` only touches the experiment manager
    through ``load_pipeline_data()`` (and only when pipeline outputs are
    enabled); every other method is used during ``run()``, which
    preflight never calls.  Returning ``None`` means "no prior pipeline
    data" — a fresh store is initialized in memory.
    """

    def load_pipeline_data(self) -> None:
        return None


@dataclass
class PreflightReport:
    """Result of validating one composed pipeline config without execution.

    Attributes:
        pipeline_name: Resolved pipeline name from the config.
        stages: Resolved stage configs (model/llm_task refs loaded).
        preload_stages: Preloaded stage name -> declared artifact keys.
        topological_order: Execution order; empty when validation failed.
        provides: Stage name -> namespaced artifact keys it produces.
        descriptions: Stage name -> one-line summary (stage class docstring).
        errors: Validation errors; empty means the pipeline would build.
    """

    pipeline_name: str
    stages: list[PipelineStageConfig] = field(default_factory=list)
    preload_stages: dict[str, list[str]] = field(default_factory=dict)
    topological_order: list[str] = field(default_factory=list)
    provides: dict[str, list[str]] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the pipeline passed every build-time check."""
        return not self.errors


def preflight_pipeline(cfg: AppConfig) -> PreflightReport:
    """Validate a composed AppConfig exactly like the executor would at build time.

    Constructs a real ``PipelineExecutor`` (with a null experiment
    manager) so the checks are the executor's own — not a reimplementation
    that could drift.  Errors are collected into the report instead of
    raised, so a multi-combo preflight can show every problem at once.

    Side effects: resolves ``cfg.pipeline.stages`` in place (same step
    ``run_pipeline`` performs) and reads preloaded artifact files from
    disk for hash/schema verification.  Never writes.

    Args:
        cfg: Fully composed AppConfig (paths resolved).

    Returns:
        PreflightReport with the DAG structure and any validation errors.
    """
    report = PreflightReport(pipeline_name=cfg.pipeline.name or "(unnamed)")

    try:
        cfg.pipeline.stages = resolve_pipeline_stages(cfg.pipeline)
    except (ValueError, TypeError, FileNotFoundError) as e:
        report.errors.append(f"stage resolution failed: {e}")
        return report

    report.stages = list(cfg.pipeline.stages)
    report.preload_stages = {
        name: list(keys) for name, keys in (cfg.pipeline.preload_stages or {}).items()
    }
    report.provides, report.descriptions = _collect_stage_meta(report.stages, cfg)

    try:
        executor = PipelineExecutor(cfg=cfg, run_manager=_NullRunManager(), data=[])
        report.topological_order = list(executor.topological_order)
    except (ValueError, TypeError, FileNotFoundError, ArtifactSchemaError) as e:
        report.errors.append(str(e))

    return report


def _own_docstring_summary(stage_cls: Callable[..., object]) -> str:
    """First line of the stage class's OWN docstring, empty if inherited.

    Reads ``cls.__doc__`` directly (not ``inspect.getdoc``): a class
    that declares no docstring has ``__doc__ is None`` — it does not
    inherit the generic ``PipelineStage`` base docstring the way normal
    attributes are inherited, which is exactly what we want.
    """
    doc = getattr(stage_cls, "__doc__", None)
    if not doc:
        return ""
    return doc.strip().split("\n", 1)[0].strip()


def _collect_stage_meta(
    stages: list[PipelineStageConfig], cfg: AppConfig
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Map each stage name to (produced artifact keys, one-line summary).

    Both derive from the resolved stage class, so they share one lookup.
    Unknown stage types are skipped here — they are already reported as
    errors by config validation / executor construction.
    """
    provides: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    for stage_cfg in stages:
        try:
            stage_cls = get_stage_fn(resolve_stage_type(stage_cfg))
            instance = stage_cls()
        except (ValueError, ImportError):
            continue
        fields = instance.get_provides_artifacts(stage_cfg, cfg=cfg)
        provides[stage_cfg.name] = [f"{stage_cfg.name}.{f}" for f in fields]
        summary = _own_docstring_summary(stage_cls)
        if summary:
            descriptions[stage_cfg.name] = summary
    return provides, descriptions


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------


def _stage_label(stage_cfg: PipelineStageConfig) -> str:
    """Human-readable stage label: name, type (if different), overrides."""
    label = stage_cfg.name
    stage_type = resolve_stage_type(stage_cfg)
    if stage_type != stage_cfg.name:
        label += f" ({stage_type})"
    extras = []
    if stage_cfg.model:
        extras.append(f"model={stage_cfg.model}")
    if stage_cfg.llm_task:
        extras.append(f"llm_task={stage_cfg.llm_task}")
    if extras:
        label += f" [{', '.join(extras)}]"
    return label


def render_dag_ascii(report: PreflightReport) -> str:
    """Render the pipeline DAG as an indented text listing.

    Stages appear in execution order when validation succeeded, else in
    declaration order (so a broken DAG is still inspectable).
    """
    lines = [f"Pipeline: {report.pipeline_name}"]

    for name, keys in sorted(report.preload_stages.items()):
        lines.append(f"  [preloaded] {name}  ~> {', '.join(keys)}")

    if report.topological_order:
        ordered = report.topological_order
        lines.append("  Execution order:")
    else:
        ordered = [s.name for s in report.stages]
        lines.append("  Stages (declaration order — no valid execution order):")

    stage_map = {s.name: s for s in report.stages}
    for i, name in enumerate(ordered, start=1):
        stage_cfg = stage_map[name]
        line = f"    {i}. {_stage_label(stage_cfg)}"
        deps = stage_cfg.depends_on or []
        if deps:
            line += f"  <- {', '.join(deps)}"
        produced = report.provides.get(name, [])
        if produced:
            line += f"  -> {', '.join(produced)}"
        lines.append(line)
        summary = report.descriptions.get(name)
        if summary:
            lines.append(f"       {summary}")

    return "\n".join(lines)


def render_dag_mermaid(report: PreflightReport) -> str:
    """Render the pipeline DAG as a Mermaid ``flowchart TD`` block.

    Preloaded stages (preloaded artifacts) appear as dashed nodes.  The
    output is valid inside a ```` ```mermaid ```` fence.
    """
    lines = ["flowchart TD"]

    for name in sorted(report.preload_stages):
        lines.append(f'    {name}(["{name} (preloaded)"])')
        lines.append(f"    style {name} stroke-dasharray: 5 5")

    for stage_cfg in report.stages:
        label = _stage_label(stage_cfg)
        summary = report.descriptions.get(stage_cfg.name)
        if summary:
            label += f"<br/><small>{summary}</small>"
        # Mermaid ["..."] labels break on a literal double quote; the
        # #quot; entity renders as one. <br/>/<small> are honored as HTML.
        label = label.replace('"', "#quot;")
        lines.append(f'    {stage_cfg.name}["{label}"]')

    for stage_cfg in report.stages:
        for dep in stage_cfg.depends_on or []:
            lines.append(f"    {dep} --> {stage_cfg.name}")

    return "\n".join(lines)
