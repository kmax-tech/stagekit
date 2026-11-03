import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from stagekit.config.classes import AppConfig, PipelineStageConfig
from stagekit.config.paths import project_path
from stagekit.models.base import BaseModel, LLMBaseModel, LLMCallResult
from stagekit.prompts.debug_utils import save_rendered_prompt_debug
from stagekit.utils.serialization import load_json, load_jsonl

logger = logging.getLogger(__name__)

# Shared constant for the annotation-flag field name used by
# mark_needs_enrichment (writer) and llm_enrich (reader).
ENRICHMENT_FLAG_FIELD = "needs_enrichment"


@dataclass
class StageResult:
    """Structured return type for pipeline stages."""
    metrics: dict
    artifacts: dict
    data: Optional[Any] = None


@dataclass
class StageContext:
    """Runner-provided context passed to every stage."""
    cfg: AppConfig
    artifacts: dict
    stage_cfg: PipelineStageConfig
    stage_dir: Path


class PipelineStage:
    """
    Base class for all pipeline stages.

    Subclasses declare their requirements via class variables:

        requires_model:      True if the stage needs a loaded model.
        requires_data:       True if the stage needs non-empty input data.
        required_artifacts:  List of artifact keys that must exist.
        required_enrichments:    List of record-level annotation keys that must
                             exist (checked against data at validation time).
        provides_enrichments:    List of record-level annotation keys this stage
                             writes.  The executor validates that every
                             ``required_enrichments`` key is provided by an
                             earlier stage in topological order.
                             Note: canonical loaders intentionally omit this —
                             they inject pre-existing enrichments from disk
                             rather than producing them via computation.
        required_config:     List of keys that MUST be in stage_cfg.config_override.
        optional_config:     List of keys the stage reads with a default.
                             required_config + optional_config (aggregated over
                             the class hierarchy) are the only keys allowed in
                             config_override — anything else is a config error,
                             so a typo cannot silently fall back to a default.

    The runner calls ``validate()`` before ``run()``.  Failures raise
    ``StageValidationError`` with a clear message naming the stage and
    what is missing.

    Contracts:

        Data: ``data`` may be a list of in-memory records **or** a path
        (str / Path) to a JSONL file.  Call ``self.resolve_data(data)``
        to normalise.

        Artifacts: all artifact keys emitted in StageResult.artifacts MUST
        be namespaced as ``"{stage_name}.{field}"``, where stage_name comes
        from ``ctx.stage_cfg.name``.  The executor merges artifacts into a
        global dict — bare keys risk silent collisions between stages.
        Consumers reference artifacts by their full namespaced key.

        Stage name: stages derive their identity from ``ctx.stage_cfg.name``
        at runtime.  Do not hardcode stage names as class attributes or
        string literals.
    """

    # ------------------------------------------------------------------
    # Requirement declarations (override in subclasses)
    # ------------------------------------------------------------------
    requires_model: bool = False
    requires_data: bool = False
    manages_own_prompts: bool = False  # True = stage sets parser/template itself, no llm_task needed
    required_artifacts: List[str] = []
    required_enrichments: List[str] = []
    provides_enrichments: List[str] = []
    provides_artifacts: List[str] = []
    # Artifacts of this stage that may cross the run boundary, i.e. be
    # pinned as another pipeline's input: {artifact_field: filename}.
    # Read by stagekit/run/pins.py (contributors: its PUBLISHER_STAGE_TYPES);
    # the filename is the artifact's name in a `pin --list`/copy, unique
    # across all publishers.
    publishes: dict = {}
    _upstream_field: str | None = None
    required_config: List[str] = []
    # table_method is legal on EVERY stage: it names the stage's row in
    # the tables and the method segment of a baseline pin coordinate.
    optional_config: List[str] = ["table_method"]
    artifact_schemas: dict = {}  # field → schema spec (see artifact_schemas.py)
    # Prompt and parser registry keys this stage uses (code-path stages
    # only — YAML-path stages get theirs from llm_task config).  Inventory
    # declarations so `uses_prompts`/`uses_parsers` + YAML llm_task refs
    # cover the full prompt/parser surface for audits and preflight.
    uses_prompts: List[str] = []
    uses_parsers: List[str] = []
    table_method: str | None = None  # if set, this stage's metrics appear in table_summary
    eval_mode: str | None = None  # "cv" or "single_run" — how metrics were computed
    # config_override keys holding filesystem paths, checked at preflight
    # (_check_stage_paths). Declared HERE, on the stage that owns the key —
    # a central list in validation.py drifted (taxonomy_path was missed).
    # input keys must exist (file or dir); output keys are created by the
    # stage, so only their parent directory must exist. get_config() hands
    # both out resolved below the project root (config.paths.project_path).
    input_path_config_keys: List[str] = []
    output_path_config_keys: List[str] = []

    # ------------------------------------------------------------------
    # Config-time validation (no data, no filesystem)
    # ------------------------------------------------------------------
    @classmethod
    def allowed_config_keys(cls) -> set[str]:
        """Every config_override key this stage type reads.

        Union of ``required_config`` and ``optional_config`` over the
        whole class hierarchy, so subclasses declare only their own
        additions.
        """
        keys: set[str] = set()
        for klass in cls.__mro__:
            keys.update(getattr(klass, "required_config", []))
            keys.update(getattr(klass, "optional_config", []))
        return keys

    @classmethod
    def input_path_keys(cls) -> set[str]:
        """config_override keys whose values must exist at preflight (MRO-union)."""
        keys: set[str] = set()
        for klass in cls.__mro__:
            keys.update(getattr(klass, "input_path_config_keys", []))
        return keys

    @classmethod
    def output_path_keys(cls) -> set[str]:
        """config_override keys this stage creates; parent dir must exist (MRO-union)."""
        keys: set[str] = set()
        for klass in cls.__mro__:
            keys.update(getattr(klass, "output_path_config_keys", []))
        return keys

    @classmethod
    def required_config_keys(cls) -> set[str]:
        """Every config_override key this stage type REQUIRES.

        Union of ``required_config`` over the whole class hierarchy —
        same rule as ``allowed_config_keys()``. Reading the bare class
        attribute instead would let a subclass's own list SHADOW the
        parent's: a subclass declaring only its additions would silently
        lose the parent's mandatory keys, preflight would pass, and the
        run would crash mid-pipeline on the missing key.
        """
        keys: set[str] = set()
        for klass in cls.__mro__:
            keys.update(getattr(klass, "required_config", []))
        return keys

    @classmethod
    def check_stage_config(cls, stage_cfg: PipelineStageConfig, cfg: AppConfig) -> list[str]:
        """Validate a stage's config before the pipeline runs.

        Called by ``validate_app_config`` for every stage whose type
        resolves to this class.  The base implementation checks
        ``required_config`` (MRO-union) and rejects keys outside
        ``allowed_config_keys()``; subclasses override to add
        stage-specific rules (extend the list returned by ``super()``).

        Returns error strings; empty list means valid.
        """
        stage_type = stage_cfg.stage_type or stage_cfg.name
        config_override = stage_cfg.config_override or {}
        errors: list[str] = []
        for key in sorted(cls.required_config_keys()):
            if key not in config_override:
                errors.append(
                    f"stage '{stage_cfg.name}': {stage_type} requires "
                    f"'{key}' in config_override"
                )
        allowed = cls.allowed_config_keys()
        for key in sorted(set(config_override) - allowed):
            errors.append(
                f"stage '{stage_cfg.name}': unknown config_override key "
                f"'{key}' for {stage_type} — declared keys: "
                f"{sorted(allowed) if allowed else '(none)'}"
            )
        return errors

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(
        self,
        *,
        stage_name: str,
        model: BaseModel | None,
        data: Any,
        ctx: StageContext,
    ) -> None:
        """Check declared requirements. Raises ``StageValidationError``."""
        errors: list[str] = []

        if self.requires_model and model is None:
            errors.append("requires a model but none was resolved")

        if self.requires_data:
            resolved = data
            if isinstance(resolved, (str, Path)):
                if not Path(resolved).exists():
                    errors.append(f"requires data but file not found: {resolved}")
            elif not resolved:
                errors.append("requires data but received empty/None")

        for key in self.required_artifacts:
            if key not in ctx.artifacts:
                errors.append(f"missing required artifact: '{key}'")

        if self.required_enrichments and isinstance(data, list) and data:
            for key in self.required_enrichments:
                present = sum(
                    1 for rec in data
                    if isinstance(rec, dict) and key in rec.get("enrichments", {})
                )
                if present == 0:
                    errors.append(
                        f"missing required annotation: '{key}' "
                        f"(0/{len(data)} records)"
                    )
                elif present < len(data):
                    # Warn but don't fail — partial annotation coverage is
                    # expected in incremental pipelines where earlier stages
                    # annotate only a subset (e.g. needs_enrichment flag).
                    logger.warning(
                        "Stage '%s': annotation '%s' present on "
                        "%d/%d records",
                        stage_name, key, present, len(data),
                    )

        upstream_field = getattr(self, "_upstream_field", None)
        if upstream_field is not None:
            deps = (ctx.stage_cfg.depends_on or []) if ctx.stage_cfg else []
            if len(deps) != 1:
                errors.append(
                    f"declares _upstream_field='{upstream_field}' but "
                    f"depends_on has {len(deps)} entries (expected exactly 1)"
                )
            else:
                key = f"{deps[0]}.{upstream_field}"
                if key not in ctx.artifacts:
                    errors.append(f"missing required artifact: '{key}'")

        config_override = self.get_config(ctx.stage_cfg)
        for key in sorted(self.required_config_keys()):
            if key not in config_override:
                errors.append(
                    f"missing required config_override key: '{key}'"
                )

        if errors:
            raise StageValidationError(stage_name, errors)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def get_config(self, stage_cfg: Optional[PipelineStageConfig] = None) -> dict:
        """The stage's config_override (empty without one), its declared path
        keys (``input_path_keys``, ``output_path_keys``) resolved below the
        project root (``project_path``) — a copy; the config keeps its text."""
        if stage_cfg is None:
            return {}
        co = dict(stage_cfg.config_override or {})
        for key in self.input_path_keys() | self.output_path_keys():
            if co.get(key):
                co[key] = str(project_path(co[key]))
        return co

    def get_provides_enrichments(
        self,
        stage_cfg: Optional[PipelineStageConfig] = None,
        cfg: Optional[AppConfig] = None,
    ) -> List[str]:
        """Return the annotation keys this stage writes.

        Override for stages where the key is determined by resolved config
        (e.g. LLMEnrichStage derives it from ``llm_task_cfg.task_name``).
        """
        return self.provides_enrichments

    def get_provides_artifacts(
        self,
        stage_cfg: Optional[PipelineStageConfig] = None,
        cfg: Optional[AppConfig] = None,
    ) -> List[str]:
        """Return the artifact field names this stage produces.

        These are bare field names (e.g. ``"raw_probs"``), not namespaced
        keys.  The executor namespaces them as ``"{stage_name}.{field}"``
        during DAG validation.
        """
        return self.provides_artifacts

    # ------------------------------------------------------------------
    # Data resolution
    # ------------------------------------------------------------------
    def resolve_data(self, data: Any) -> Any:
        """Normalise *data* to in-memory records when it is a file reference."""
        if data is None:
            return []
        if isinstance(data, (str, Path)):
            p = Path(data)
            if p.suffix == ".json":
                return load_json(p)
            return load_jsonl(p)
        return data

    # ------------------------------------------------------------------
    # Pipeline data helpers
    # ------------------------------------------------------------------
    def publish_output(self, ctx: StageContext, stage_name: str, field: str, values) -> None:
        """Record output in the pipeline data store, if enabled."""
        pipeline_data = ctx.artifacts.get("pipeline_data")
        if pipeline_data:
            pipeline_data.add_output(stage_name=stage_name, field=field, values=values)

    def resolve_upstream_artifact(self, ctx: StageContext) -> Any:
        """Return the value of the upstream artifact declared by ``_upstream_field``.

        Safe to call after ``validate()`` has passed.
        """
        deps = ctx.stage_cfg.depends_on
        assert deps, f"resolve_upstream_artifact requires depends_on (stage: {ctx.stage_cfg.name})"
        key = f"{deps[0]}.{self._upstream_field}"
        return ctx.artifacts[key]

    # ------------------------------------------------------------------
    # Main entry point (called by the runner, overridden by subclasses)
    # ------------------------------------------------------------------
    def run(
        self,
        model: BaseModel | None,
        data: Any,
        ctx: StageContext,
    ) -> StageResult:
        """Execute the stage.

        Args:
            model: Resolved model instance, or None for model-free stages.
            data: List of records or a path to a JSON/JSONL file —
                normalise via ``self.resolve_data(data)``.
            ctx: Runner-provided context (config, artifacts, stage config,
                stage output directory).

        Returns:
            StageResult with metrics, namespaced artifacts, and the
            (possibly replaced) data list.
        """
        raise NotImplementedError("Subclasses must implement run()")


# ------------------------------------------------------------------
# ID-join helper — one vocabulary for artifact↔record joins
# ------------------------------------------------------------------

@dataclass
class JoinReport:
    """Coverage report for an id-based artifact↔records join.

    Attach to StageResult.metrics via ``as_metrics()`` so coverage gaps
    surface in run summaries instead of only in logs.
    """

    label: str
    n_reference: int              # reference ids considered
    n_entries: int                # entries offered by the artifact
    n_matched: int                # ids present on both sides
    unmatched_ids: list           # reference ids without an entry (first 10)

    @property
    def coverage(self) -> float:
        return self.n_matched / self.n_reference if self.n_reference else 1.0

    def as_metrics(self, prefix: str) -> dict:
        return {
            f"{prefix}_matched": self.n_matched,
            f"{prefix}_reference": self.n_reference,
            f"{prefix}_coverage": round(self.coverage, 4),
        }


def join_by_id(
    reference_ids,
    entries,
    *,
    key: str = "id",
    on_missing: str = "drop",
    label: str = "join",
) -> tuple[dict, JoinReport]:
    """Join artifact entries to reference record ids — explicit semantics.

    *entries* is either a list of dicts carrying *key* (duplicate ids:
    last wins) or an already-built ``{id: value}`` mapping.

    on_missing controls reference ids without a matching entry:
        "drop"  (default) exclude them; a warning reports the gap
        "fail"  raise ValueError naming the unmatched ids

    Returns ``({id: entry} in reference order, JoinReport)``.  Coverage
    is always computed — use ``report.as_metrics(prefix)`` to publish it.
    """
    if on_missing not in ("drop", "fail"):
        raise ValueError(
            f"on_missing='{on_missing}' is not valid (expected 'drop' or 'fail')"
        )

    if isinstance(entries, dict):
        entry_by_id = entries
    else:
        entry_by_id = {}
        for entry in entries:
            eid = entry.get(key)
            if eid is not None:
                entry_by_id[eid] = entry

    reference_ids = [rid for rid in reference_ids if rid is not None]
    joined = {rid: entry_by_id[rid] for rid in reference_ids if rid in entry_by_id}
    unmatched = [rid for rid in reference_ids if rid not in entry_by_id]

    report = JoinReport(
        label=label,
        n_reference=len(reference_ids),
        n_entries=len(entry_by_id),
        n_matched=len(joined),
        unmatched_ids=unmatched[:10],
    )

    if unmatched:
        if on_missing == "fail":
            raise ValueError(
                f"{label}: {len(unmatched)}/{len(reference_ids)} reference "
                f"id(s) have no matching entry (first 5: {unmatched[:5]}). "
                f"Set on_missing='drop' to tolerate."
            )
        logger.warning(
            "%s: %d/%d reference id(s) without matching entry — dropped "
            "(coverage %.1f%%, first 5: %s)",
            label, len(unmatched), len(reference_ids),
            100 * report.coverage, unmatched[:5],
        )

    return joined, report


# ------------------------------------------------------------------
# Shared helpers for LLM-based stages
# ------------------------------------------------------------------

def build_llm_tasks(parser, data, dataset_id: str) -> tuple[list[dict], list]:
    """Extract fields from records and build LLM task dicts.

    Returns (tasks, record_ids) where each task has parser fields,
    item_id, and dataset_id ready for run_llm_call / run_llm_tasks.
    """
    tasks = []
    record_ids = []
    for record in data:
        fields = parser.extract_fields_from_record(record)
        item_id = record.get("id")
        record_ids.append(item_id)
        tasks.append({
            **fields,
            "item_id": str(item_id) if item_id is not None else None,
            "dataset_id": dataset_id,
        })
    return tasks, record_ids


def run_llm_batched(model, tasks: list[dict], *, batch_size: int, desc: str) -> list:
    """Run LLM tasks in batches, returning a flat list of LLMCallResult."""
    all_results = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        results = model.run_llm_tasks(
            tasks=batch,
            desc=f"{desc} [{start}:{start + len(batch)}]",
        )
        all_results.extend(results)
    return all_results


def run_llm_batched_with(
    model, tasks: list[dict], *, template, parser, batch_size: int, desc: str,
) -> list:
    """Run LLM tasks in batches with explicit template/parser (no model mutation)."""
    all_results = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        results = model.run_llm_tasks_with(
            template=template,
            parser=parser,
            tasks=batch,
            desc=f"{desc} [{start}:{start + len(batch)}]",
        )
        all_results.extend(results)
    return all_results


class LLMBatchStage(PipelineStage):
    """Shared core for stages that run batch LLM inference over records.

    Owns the one canonical flow:

        select_records → build_tasks → debug_prompt → run_llm_batched
        → length check → emit_results

    Subclasses choose the OUTPUT CONVENTION by implementing
    ``emit_results``:

        LLMGenerateStage — results become a namespaced artifact
                           (``{stage_name}.{output_field}``), records
                           stay untouched.
        LLMEnrichStage   — results are written into the records as
                           enrichments (data channel), no artifacts.

    Selection/task hooks let subclasses filter records
    (``needs_enrichment`` flags, upstream joins) or add task fields
    without re-implementing the batching loop.
    """

    requires_model = True
    requires_data = True

    desc: str
    default_batch_size = 300

    optional_config = ["batch_size"]

    def validate(self, *, stage_name, model, data, ctx):
        super().validate(stage_name=stage_name, model=model, data=data, ctx=ctx)
        if model is not None and not getattr(model, "parser", None):
            raise StageValidationError(stage_name, [
                "requires a model with a parser (set llm_task on the stage or globally)"
            ])

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def select_records(self, data, co: dict, ctx: StageContext) -> tuple[list[tuple[int, dict]], dict]:
        """Choose which records receive LLM calls.

        Returns ((index, record) pairs, extra_metrics).  Default: all.
        """
        return list(enumerate(data)), {}

    def cache_dataset_id(self, ctx: StageContext) -> str:
        """First path segment of the LLM cache for this stage's calls."""
        return ctx.cfg.meta.experiment_name

    def task_fields(self, model, record: dict, ctx: StageContext) -> dict:
        """Prompt fields for one record.  Default: parser extraction."""
        return model.parser.extract_fields_from_record(record)

    def build_tasks(self, model, selected: list[tuple[int, dict]], co: dict, ctx: StageContext) -> tuple[list[dict], list]:
        """Build LLM task dicts. Returns (tasks, record_ids)."""
        dataset_id = self.cache_dataset_id(ctx)
        tasks: list[dict] = []
        record_ids: list = []
        for _, record in selected:
            fields = self.task_fields(model, record, ctx)
            rid = record.get("id")
            record_ids.append(rid)
            tasks.append({
                **fields,
                "item_id": str(rid) if rid is not None else None,
                "dataset_id": dataset_id,
            })
        return tasks, record_ids

    def progress_desc(self, ctx: StageContext) -> str:
        return self.desc

    def debug_prompt(self, model, records: list[dict], tasks: list[dict], ctx: StageContext) -> None:
        """Save template + one rendered example for debugging."""
        if getattr(model, "template", None) is None or model.parser is None:
            return
        fields = model.parser.extract_fields_from_record(records[0])
        fields = model.parser.prepare_fields(fields)
        save_rendered_prompt_debug(
            template=model.template,
            fields=fields,
            output_dir=ctx.stage_dir,
        )

    def emit_results(self, *, data, selected, record_ids, results, extra_metrics, co, ctx) -> StageResult:
        """Shape LLM results into a StageResult (output convention)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------
    def run(self, model: LLMBaseModel | None, data, ctx: StageContext):  # type: ignore[override]
        assert model is not None, f"{type(self).__name__}: requires a model"
        data = self.resolve_data(data)
        stage_name = ctx.stage_cfg.name
        co = self.get_config(ctx.stage_cfg)
        batch_size = int(co.get("batch_size", self.default_batch_size))

        selected, extra_metrics = self.select_records(data, co, ctx)
        tasks, record_ids = self.build_tasks(model, selected, co, ctx)

        if tasks:
            try:
                self.debug_prompt(model, [r for _, r in selected], tasks, ctx)
            except Exception as e:
                logger.warning("%s: prompt debug rendering failed: %s", stage_name, e)

        results = run_llm_batched(
            model, tasks, batch_size=batch_size, desc=self.progress_desc(ctx),
        )

        if len(results) != len(record_ids):
            raise RuntimeError(
                f"{self.progress_desc(ctx)}: backend returned {len(results)} "
                f"results for {len(record_ids)} inputs"
            )

        return self.emit_results(
            data=data,
            selected=selected,
            record_ids=record_ids,
            results=results,
            extra_metrics=extra_metrics,
            co=co,
            ctx=ctx,
        )


class LLMGenerateStage(LLMBatchStage):
    """LLM batch stage whose results become a namespaced artifact.

    Subclasses set two class attributes and implement ``format_result``:

        output_field:  field name for artifact key and pipeline_data
        desc:          progress-bar label for run_llm_batched

    The stage name is derived from ctx.stage_cfg.name at runtime.
    """

    output_field: str

    def get_provides_artifacts(self, stage_cfg=None, cfg=None):
        return [self.output_field]

    def format_result(self, index: int, record_id: Any, result: LLMCallResult) -> dict:
        """Shape a single LLM result into the output dict."""
        raise NotImplementedError

    def validate_outputs(self, outputs: list[dict], stage_name: str) -> None:
        """Optional schema check on the formatted outputs (default: none)."""

    def emit_results(self, *, data, selected, record_ids, results, extra_metrics, co, ctx) -> StageResult:
        stage_name = ctx.stage_cfg.name

        outputs = []
        dropped = 0
        for i, (rid, r) in enumerate(zip(record_ids, results)):
            if not r.success or r.value is None:
                dropped += 1
                continue
            outputs.append(self.format_result(i, rid, r))

        if dropped:
            logger.warning(
                "%s: dropped %d/%d results (LLM returned invalid output)",
                self.desc, dropped, len(record_ids),
            )

        self.validate_outputs(outputs, stage_name)
        self.publish_output(ctx, stage_name, self.output_field, outputs)

        return StageResult(
            metrics={
                "count": len(outputs),
                "dropped_invalid": dropped,
                **extra_metrics,
                **LLMCallResult.summarize(results),
            },
            artifacts={f"{stage_name}.{self.output_field}": outputs},
            data=data,
        )


class LLMWorkflowStage(PipelineStage):
    """Base for stages that run a SEQUENCE of LLM steps with data flowing
    between them (multi-step workflows, e.g. generate → edit).

    Each step is built from the same registered units as single-step
    llm_tasks: a prompt from PROMPT_REGISTRY and a parser from
    PARSER_REGISTRY, resolved by name in ``run_llm_step``.  The chaining
    logic — feeding one step's results into the next step's tasks — stays
    plain Python in the subclass's ``run()``; the pipeline DAG is the
    layer for coarse steps, not for intra-stage wiring.
    """

    requires_model = True
    manages_own_prompts = True

    optional_config = ["batch_size"]

    def run_llm_step(
        self,
        model,
        *,
        prompt: str,
        parser: str,
        tasks: list[dict],
        ctx: StageContext,
        label: str | None = None,
        batch_size: int | None = None,
    ) -> list[LLMCallResult]:
        """Run one workflow step over *tasks*.

        Resolves *prompt*/*parser* from the registries, fills in the
        cache scope (``dataset_id``), saves a rendered prompt-debug
        example, and runs the batched stateless LLM call.  Tasks must
        carry ``item_id``; template fields come from the task dicts.
        """
        from stagekit.parsers.registry import get_parser
        from stagekit.prompts.registry import get_prompt

        stage_name = ctx.stage_cfg.name
        template = get_prompt(prompt)
        parser_obj = get_parser(parser)
        label = label or prompt

        if batch_size is None:
            batch_size = int(self.get_config(ctx.stage_cfg).get("batch_size", 300))

        dataset_id = ctx.cfg.meta.experiment_name
        tasks = [
            task if "dataset_id" in task else {**task, "dataset_id": dataset_id}
            for task in tasks
        ]

        if tasks:
            try:
                save_rendered_prompt_debug(
                    template=template,
                    fields=tasks[0],
                    output_dir=ctx.stage_dir,
                    label=label,
                )
            except Exception as e:
                logger.warning(
                    "%s: prompt debug rendering failed (%s): %s",
                    stage_name, label, e,
                )

        return run_llm_batched_with(
            model, tasks,
            template=template, parser=parser_obj,
            batch_size=batch_size,
            desc=f"{stage_name}:{label}",
        )


def set_enrichment(record: dict, key: str, value: dict) -> None:
    """Set a record-scoped annotation, creating the 'enrichments' dict if needed."""
    record.setdefault("enrichments", {})
    record["enrichments"][key] = value


def resolve_annotation(record: dict, chain: list[str]) -> dict:
    """Walk *chain* of annotation keys and return the first non-empty result.

    Each key is looked up as ``enrichments.<key>.result``; the first one
    that yields a truthy dict wins.  Returns ``{}`` if nothing matches.

    Usage::

        source = resolve_annotation(record, [
            "clean_circularity", "clean_meta", source_annotation,
        ])
        arguments = source.get("arguments", [])
    """
    ann = record.get("enrichments", {})
    for key in chain:
        result = ann.get(key, {}).get("result", {})
        if result:
            return result
    return {}


class StageValidationError(Exception):
    """Raised when a stage's declared requirements are not met."""

    def __init__(self, stage_name: str, errors: list[str]):
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        super().__init__(
            f"Stage '{stage_name}' failed validation:\n{bullet_list}"
        )
        self.stage_name = stage_name
        self.errors = errors
