import inspect
import logging
from collections import defaultdict, deque
from typing import Any

from stagekit.config.classes import AppConfig
from stagekit.models.registry import load_model
from stagekit.models.base import LLMBaseModel
from stagekit.pipeline.data_store import PipelineDataStore
from stagekit.pipeline.registry import get_stage_fn
from stagekit.stages.base import PipelineStage, StageContext, StageResult
from stagekit.config.validation import (
    check_stage_dependencies,
    check_unique_stage_names,
    resolve_stage_type,
)

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Unified DAG executor for pipeline stages (data creation, enrichment, evaluation).
    """

    def __init__(self, cfg: AppConfig, run_manager, data: Any, artifacts=None):
        if run_manager is None:
            raise ValueError("PipelineExecutor requires run_manager=RunManager")

        self.cfg = cfg
        self.run_manager = run_manager
        self.data = data

        # ------------------------------------------------------------
        # Stage configs — read from cfg.pipeline.stages
        # ------------------------------------------------------------
        if hasattr(cfg, "pipeline") and hasattr(cfg.pipeline, "stages"):
            self.stage_configs = cfg.pipeline.stages
        else:
            self.stage_configs = []

        if not self.stage_configs:
            logger.warning("No stages defined in configuration.")

        # ------------------------------------------------------------
        # Global fallback model
        # ------------------------------------------------------------
        self.global_model_cfg = cfg.model
        self.global_llm_task_cfg = cfg.llm_task

        # ------------------------------------------------------------
        # Runtime artifacts
        # ------------------------------------------------------------
        self.artifacts = artifacts or {}

        # Pre-load artifacts from pinned runs (classify-only pipelines)
        self._apply_preload_artifacts()

        # Resolve preloaded stages — explicit DAG nodes from config
        vs = self.cfg.pipeline.preload_stages or {}
        self.preload_stage_names: set[str] = set(vs)
        self.preload_stage_artifacts: set[str] = {
            key for keys in vs.values() for key in keys
        }
        self._validate_preload_stages()

        # Load or initialize pipeline data
        if self.cfg.meta.enable_pipeline_outputs:
            self._load_or_init_pipeline_data()

        # ------------------------------------------------------------
        # DAG build & validation
        # ------------------------------------------------------------
        self.stage_map = {s.name: s for s in self.stage_configs}
        self._validate_unique_names()
        self._validate_dependencies()
        self.topological_order = self._build_topological_order()
        self._validate_enrichment_contracts()
        self._validate_artifact_contracts()
        self._validate_prompt_parser_declarations()

    # ==================================================================
    # Pipeline Data Store
    # ==================================================================
    def _load_or_init_pipeline_data(self):
        # Always fresh: every run gets its own timestamped run_dir, so a
        # restore-from-disk branch here could never fire (the resume
        # subsystem it belonged to was removed — audit C3).
        self.artifacts["pipeline_data"] = PipelineDataStore()

    # ==================================================================
    # Preload artifacts from pinned runs
    # ==================================================================
    def _apply_preload_artifacts(self):
        """Load artifacts declared in pipeline.preload_artifacts from pinned runs.

        Every value is ``pin:<coordinate>``; the pin (published/pins.yaml,
        stagekit/run/pins.py) names the run, and the artifact key is read from
        that run, hash-verified against the pin, then shape-checked
        against its registered schema and merged into self.artifacts so
        downstream stages see it exactly like a pipeline-produced one.

        Side effects: ``self.preload_pins`` (coordinate → Pin) feeds the
        run's ``preload_provenance.json`` and ``preload_expect``.
        """
        from stagekit.run import pins
        from stagekit.stages.artifact_schemas import validate_preloaded_artifact

        self.preload_pins: dict[str, pins.Pin] = {}
        preload = self.cfg.pipeline.preload_artifacts
        if not preload:
            return

        for artifact_key, spec in preload.items():
            coordinate = pins.coordinate_from_spec(spec, artifact_key=artifact_key)
            pin = self.preload_pins.get(coordinate) or pins.resolve(coordinate)
            data = pins.load(pin, artifact_key)

            if not isinstance(data, (list, dict)):
                raise TypeError(
                    f"preload_artifacts: '{artifact_key}' must be a list or dict, "
                    f"got {type(data).__name__} from pin '{coordinate}'"
                )

            # Shape check against the field's registered schema — the
            # hash proves integrity, this proves the format still
            # matches today's code (cross-run drift fails here, not
            # deep inside downstream feature code).
            source = f"pin:{coordinate} -> {pin.run_dir}/{pin.artifacts[artifact_key].file}"
            validate_preloaded_artifact(artifact_key, data, source=source)

            self.artifacts[artifact_key] = data
            self.preload_pins[coordinate] = pin
            logger.info(
                "Preloaded artifact '%s' from pin '%s' (run %s, %s entries)",
                artifact_key, coordinate, pin.run_id,
                len(data) if isinstance(data, list) else len(data.keys()),
            )

        logger.info(
            "Preloaded %d artifact(s) from %d pin(s), hashes verified",
            len(preload), len(self.preload_pins),
        )
        self._validate_preload_expectations()
        self._apply_derived_artifacts()

    def _apply_derived_artifacts(self):
        """Materialise pipeline.derived_artifacts into self.artifacts.

        Each declared key is computed from its (preloaded) source with the
        named transform and shape-checked like a preload. This run owns
        the derivation; `run_inputs.yaml` records the declaration.
        """
        from stagekit.stages.derived_artifacts import derive, parse_derived_artifacts

        for key, decl in parse_derived_artifacts(self.cfg.pipeline).items():
            if decl.source not in self.artifacts:
                raise ValueError(
                    f"derived_artifacts['{key}']: source '{decl.source}' was not "
                    f"preloaded (preloaded: {sorted(self.artifacts)})"
                )
            self.artifacts[key] = derive(
                decl, self.artifacts[decl.source], source_label=decl.source,
            )

    def _validate_preload_expectations(self):
        """Hard-validate preload provenance against pipeline.preload_expect.

        Every expectation is held against EVERY pin the pipeline loaded
        from; the pipeline cannot start on a mismatch.

        Supported keys:
            run_id          — exact match against the pin's run_id
            repro_hash      — exact match against the pin's repro_hash
            artifact_hashes — {artifact_key: sha256} against the pinned hashes
        """
        expect = self.cfg.pipeline.preload_expect
        if not expect:
            return

        unknown = sorted(set(expect) - {"run_id", "repro_hash", "artifact_hashes"})
        if unknown:
            raise ValueError(
                f"preload_expect: unknown key(s) {unknown}; "
                f"supported: run_id, repro_hash, artifact_hashes"
            )
        if not self.preload_pins:
            raise ValueError(
                "preload_expect is configured but no pin was loaded "
                "(preload_artifacts is empty)"
            )

        for coordinate, pin in self.preload_pins.items():
            errors: list[str] = []
            for scalar in ("run_id", "repro_hash"):
                if scalar in expect and getattr(pin, scalar) != expect[scalar]:
                    errors.append(
                        f"{scalar}: expected '{expect[scalar]}', got '{getattr(pin, scalar)}'"
                    )
            for key, expected_hash in (expect.get("artifact_hashes") or {}).items():
                art = pin.artifacts.get(key)
                if art is None:
                    errors.append(f"artifact_hashes: '{key}' is not pinned under '{coordinate}'")
                elif art.sha256 != expected_hash:
                    errors.append(
                        f"artifact_hashes: '{key}' hash mismatch — expected "
                        f"'{expected_hash[:16]}…', pinned '{art.sha256[:16]}…'"
                    )
            if errors:
                bullet_list = "\n  - ".join(errors)
                raise ValueError(
                    f"Preloaded artifact provenance mismatch "
                    f"(pin '{coordinate}'):\n  - {bullet_list}"
                )

        logger.info("Preload expectations validated successfully")

    # ==================================================================
    # Preloaded stage validation
    # ==================================================================
    def _validate_preload_stages(self):
        """Cross-validate preload_stages against loaded artifacts.

        Every artifact key declared in preload_stages must be present
        in self.artifacts (loaded via preload_artifacts or passed
        directly).  Preloaded stage names must not clash with real stages.
        """
        preload_stages = self.cfg.pipeline.preload_stages
        if not preload_stages:
            return

        real_names = {s.name for s in self.stage_configs}
        errors: list[str] = []

        for vs_name, provides in preload_stages.items():
            if vs_name in real_names:
                errors.append(
                    f"preload_stages: '{vs_name}' conflicts with a "
                    f"real pipeline stage of the same name"
                )
            for artifact_key in provides:
                if artifact_key not in self.artifacts:
                    errors.append(
                        f"preload_stages: '{vs_name}' declares artifact "
                        f"'{artifact_key}' but it was not preloaded "
                        f"(check preload_artifacts)"
                    )

        if errors:
            bullet_list = "\n  - ".join(errors)
            raise ValueError(
                f"Preloaded stage validation failed:\n  - {bullet_list}"
            )

    # ==================================================================
    # DAG validation
    # ==================================================================
    def _validate_unique_names(self):
        errors = check_unique_stage_names(self.stage_configs)
        if errors:
            raise ValueError(errors[0])

    def _validate_dependencies(self):
        errors = check_stage_dependencies(
            self.stage_configs, self.preload_stage_names
        )
        if errors:
            raise ValueError(errors[0])

    def _validate_enrichment_contracts(self):
        """Check that every required_enrichments key is provided by an earlier stage."""
        available: set[str] = set()

        for stage_name in self.topological_order:
            stage_cfg = self.stage_map[stage_name]
            stage_type = resolve_stage_type(stage_cfg)
            stage_cls = get_stage_fn(stage_type)
            instance = stage_cls()

            for key in instance.required_enrichments:
                if key not in available:
                    raise ValueError(
                        f"Stage '{stage_name}' requires enrichment '{key}' "
                        f"but no earlier stage provides it. "
                        f"Available: {sorted(available) if available else '(none)'}"
                    )

            available.update(instance.get_provides_enrichments(stage_cfg, cfg=self.cfg))

    def _validate_artifact_contracts(self):
        """Check that every _upstream_field requirement is satisfiable at DAG build time."""
        # Seed with artifact keys declared by preloaded stages.
        available: set[str] = set(self.preload_stage_artifacts)

        for stage_name in self.topological_order:
            stage_cfg = self.stage_map[stage_name]
            stage_type = resolve_stage_type(stage_cfg)
            stage_cls = get_stage_fn(stage_type)
            instance = stage_cls()

            # Check _upstream_field requirement
            upstream_field = getattr(instance, "_upstream_field", None)
            if upstream_field is not None:
                deps = stage_cfg.depends_on or []
                if len(deps) != 1:
                    raise ValueError(
                        f"Stage '{stage_name}' declares _upstream_field='{upstream_field}' "
                        f"but has {len(deps)} dependencies (exactly 1 required)."
                    )
                key = f"{deps[0]}.{upstream_field}"
                if key not in available:
                    raise ValueError(
                        f"Stage '{stage_name}' requires artifact '{key}' "
                        f"but stage '{deps[0]}' does not provide field "
                        f"'{upstream_field}'. "
                        f"Available: {sorted(available) if available else '(none)'}"
                    )

            # Register what this stage provides
            for field in instance.get_provides_artifacts(stage_cfg, cfg=self.cfg):
                available.add(f"{stage_name}.{field}")

    def _validate_prompt_parser_declarations(self):
        """Check that every uses_prompts/uses_parsers key resolves in its registry."""
        from stagekit.prompts.registry import PROMPT_REGISTRY
        from stagekit.parsers.registry import PARSER_REGISTRY

        for stage_name in self.topological_order:
            stage_cfg = self.stage_map[stage_name]
            stage_type = resolve_stage_type(stage_cfg)
            stage_cls = get_stage_fn(stage_type)

            for prompt_key in stage_cls.uses_prompts:
                try:
                    PROMPT_REGISTRY.get(prompt_key)
                except KeyError:
                    raise ValueError(
                        f"Stage '{stage_name}' ({stage_type}) declares "
                        f"uses_prompts=[..., '{prompt_key}', ...] but "
                        f"'{prompt_key}' is not in PROMPT_REGISTRY"
                    )

            for parser_key in stage_cls.uses_parsers:
                try:
                    PARSER_REGISTRY.get(parser_key)
                except KeyError:
                    raise ValueError(
                        f"Stage '{stage_name}' ({stage_type}) declares "
                        f"uses_parsers=[..., '{parser_key}', ...] but "
                        f"'{parser_key}' is not in PARSER_REGISTRY"
                    )

    # ==================================================================
    # DAG build
    # ==================================================================
    def _build_topological_order(self):
        real_names = {s.name for s in self.stage_configs}

        graph = defaultdict(list)
        indegree = defaultdict(int)

        for stage in self.stage_configs:
            name = stage.name
            indegree.setdefault(name, 0)

            for dep in stage.depends_on or []:
                graph[dep].append(name)
                indegree[name] += 1
                # Preloaded deps enter the graph as zero-indegree nodes
                # so the topological sort resolves correctly.
                if dep in self.preload_stage_names:
                    indegree.setdefault(dep, 0)

        queue = deque([s for s, deg in indegree.items() if deg == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in graph[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        # Strip preloaded stages — they don't execute, only unblock deps
        real_order = [s for s in order if s in real_names]

        if len(real_order) != len(self.stage_configs):
            raise ValueError("Cycle detected in pipeline DAG")

        logger.info("Execution order: %s", " -> ".join(real_order))
        return real_order

    # ==================================================================
    # Per-stage model resolution
    # ==================================================================
    def get_model_for_stage(self, stage_cfg):
        """
        Resolve the model for a stage.

        Returns (model, model_cfg, llm_task_cfg).  All three are None when
        the stage runs without a model.

        Resolution order:
          1. Stage-level model + llm_task override (both set)
          2. Stage-level model override + global llm_task
          3. Global model + stage-level llm_task override
          4. Global model + global llm_task (if global model_type is set)
          5. (None, None, None) — stage runs without a model
        """
        stage_model_cfg = stage_cfg.model_cfg
        stage_llm_task_cfg = stage_cfg.llm_task_cfg

        if stage_model_cfg and stage_llm_task_cfg:
            model_cfg, llm_task_cfg = stage_model_cfg, stage_llm_task_cfg
        elif stage_model_cfg:
            model_cfg, llm_task_cfg = stage_model_cfg, self.global_llm_task_cfg
        elif stage_llm_task_cfg:
            model_cfg, llm_task_cfg = self.global_model_cfg, stage_llm_task_cfg
        elif (
            self.global_model_cfg
            and self.global_model_cfg.model_type
            and self.global_model_cfg.backend
        ):
            model_cfg, llm_task_cfg = self.global_model_cfg, self.global_llm_task_cfg
        else:
            return None, None, None

        return load_model(model_cfg, llm_task_cfg, self.cfg.meta), model_cfg, llm_task_cfg

    # ==================================================================
    # LLM diagnostics
    # ==================================================================
    def _save_llm_diagnostics(self, stage_name: str, model):
        """Persist LLM stats and parse diagnostics for a completed stage."""
        if not model or not hasattr(model, "llm_stats"):
            return

        stats = model.llm_stats
        self.run_manager.save_llm_stage_stats(stage_name, stats)

        if isinstance(model, LLMBaseModel) and stats.parse_diagnostics:
            # ALL parser buckets, not just the model's default parser:
            # workflow stages pass their own registry parser per step, so
            # their diagnostics live under other cache_ids — reading only
            # model.parser would silently drop every one of them. Each
            # entry carries its parser_id, so the merge is lossless.
            merged = [
                d
                for entries in stats.parse_diagnostics.values()
                for d in entries
            ]
            self.run_manager.save_parse_diagnostics(stage_name, merged)

        if stats.retry_exhausted_items:
            total = sum(len(v) for v in stats.retry_exhausted_items.values())
            logger.warning(
                "Stage '%s' completed with %d failed items after retries",
                stage_name, total,
            )


    # ==================================================================
    # Table summary
    # ==================================================================
    def _build_table_entry(
        self, method: str, stage: str, metrics: dict, *,
        eval_mode: str | None = None, model_cfg=None,
    ) -> dict:
        """Build a normalized table_summary entry from run config + stage metrics.

        Contract: see TABLE_SUMMARY_* constants in artifact_schemas.py.

        Args:
            model_cfg: The model THIS stage actually ran with (stage-level
                ``model:`` overrides included). Falls back to the global
                model — reading only the global one mislabeled every table
                row produced by a stage with its own model override.
        """
        from stagekit.stages.artifact_schemas import TABLE_SUMMARY_METRIC_KEYS

        cfg = self.cfg
        # Keep only recognised metric names (+ their _std). Bare key = fold
        # mean everywhere since the {metric}_mean vocabulary was retired.
        clean_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if k in TABLE_SUMMARY_METRIC_KEYS:
                clean_metrics[k] = v
            elif k.endswith("_std"):
                base = k.removesuffix("_std")
                if base in TABLE_SUMMARY_METRIC_KEYS:
                    clean_metrics[f"{base}_std"] = v

        effective_model = model_cfg if model_cfg is not None else cfg.model
        entry = {
            "method": method,
            "stage": stage,
            "dataset": cfg.data.data_name,
            "model": effective_model.model_name if effective_model else None,
            "mode": cfg.pipeline.mode or None,
            "variant": cfg.pipeline.variant or None,
            "source_model": cfg.pipeline.source_model or None,
            "eval_mode": eval_mode,
            "metrics": clean_metrics,
        }
        if metrics.get("n_folds") is not None:
            entry["n_folds"] = metrics["n_folds"]
        return entry

    # ==================================================================
    # Single stage execution
    # ==================================================================
    def _execute_stage(self, stage_name, stage_cfg, stage_fn, model) -> tuple[StageResult, tuple[str, str | None] | None]:
        """Run a single stage: validate, execute, persist.

        Returns (result, table_method).  table_method is non-None only for
        stages that declare themselves as the primary metric source for
        table_summary (e.g. eval_predictions, svm_clf, bert_text_clf).
        """
        stage_dir = self.run_manager.stage_dir(stage_name)

        ctx = StageContext(
            cfg=self.cfg,
            artifacts=self.artifacts,
            stage_cfg=stage_cfg,
            stage_dir=stage_dir,
        )

        if not (inspect.isclass(stage_fn) and issubclass(stage_fn, PipelineStage)):
            raise TypeError(
                f"Stage '{stage_name}' (impl: {stage_fn}) is not a valid PipelineStage class."
            )

        stage_instance = stage_fn()
        stage_instance.validate(stage_name=stage_name, model=model, data=self.data, ctx=ctx)
        result = stage_instance.run(model=model, data=self.data, ctx=ctx)

        if not isinstance(result, StageResult):
            raise TypeError(
                f"Stage '{stage_name}' must return a StageResult, got {type(result).__name__}."
            )

        # Validate artifact schemas (completeness + shape)
        from stagekit.stages.artifact_schemas import validate_stage_artifacts
        validate_stage_artifacts(stage_name, result, stage_instance, stage_cfg, self.cfg)

        # Persist through RunManager
        self.run_manager.save_stage_outputs(stage_name, result.metrics, result.artifacts)
        self.run_manager.mark_stage_completed(stage_name)
        table_info = None
        tm = getattr(stage_instance, "table_method", None)
        if tm:
            table_info = (tm, getattr(stage_instance, "eval_mode", None))
        return result, table_info

    # ==================================================================
    # RUN PIPELINE
    # ==================================================================
    def run(self):
        """Runs pipeline in topological order."""
        all_metrics = {}
        table_entries: list[dict] = []
        # Read by the caller's failure handler: which stages finished
        # before a crash, for the "failed" pipeline_status.json.
        self.completed_stages: list[str] = []

        # One-shot: save reproducibility snapshot
        self.run_manager.save_config_snapshot()

        # Save preload provenance (the pins loaded from) next to the snapshot
        if self.preload_pins:
            from stagekit.utils.serialization import save_json as _save_json
            _save_json(
                {c: pin.to_dict() for c, pin in self.preload_pins.items()},
                self.run_manager.run_dir / "preload_provenance.json",
            )

        for stage_name in self.topological_order:
            stage_cfg = self.stage_map[stage_name]

            # --------------------------------------------------
            # Resolve function from stage_type or name
            # --------------------------------------------------
            stage_type = resolve_stage_type(stage_cfg)

            try:
                stage_fn = get_stage_fn(stage_type)
            except ValueError as e:
                raise ValueError(
                    f"Could not resolve stage type: '{stage_type}' for stage '{stage_cfg.name}'. "
                    f"Ensure it is registered via @register_stage."
                ) from e

            # --------------------------------------------------
            # Resolve model (or None)
            # --------------------------------------------------
            model, model_cfg, llm_task_cfg = self.get_model_for_stage(stage_cfg)

            # --------------------------------------------------
            # Record per-stage execution
            # --------------------------------------------------
            self.run_manager.record_stage_execution(
                stage_name=stage_cfg.name,
                stage_cfg=stage_cfg,
                model_cfg=model_cfg,
                llm_task_cfg=llm_task_cfg,
                model=model,
            )

            # --------------------------------------------------
            # Reset LLM stats
            # --------------------------------------------------
            if isinstance(model, LLMBaseModel):
                model.llm_stats.reset()

            # --------------------------------------------------
            # Execute stage
            # --------------------------------------------------
            result, table_info = self._execute_stage(
                stage_name=stage_name,
                stage_cfg=stage_cfg,
                stage_fn=stage_fn,
                model=model,
            )

            # --------------------------------------------------
            # Update data if stage returned it
            # --------------------------------------------------
            if result.data is not None:
                if result.data is not self.data:
                    logger.info("Stage '%s' updated the dataset.", stage_name)
                self.data = result.data

            # --------------------------------------------------
            # LLM diagnostics
            # --------------------------------------------------
            self._save_llm_diagnostics(stage_name, model)

            # --------------------------------------------------
            # Merge artifacts
            # --------------------------------------------------
            self.artifacts.update(result.artifacts)

            # Persist pipeline data
            if "pipeline_data" in self.artifacts:
                self.run_manager.save_pipeline_data(self.artifacts["pipeline_data"])

            all_metrics[stage_name] = result.metrics

            # --------------------------------------------------
            # Collect table_summary entry if stage declares table_method
            # --------------------------------------------------
            if table_info:
                table_method, eval_mode = table_info
                table_entries.append(self._build_table_entry(
                    method=table_method,
                    stage=stage_name,
                    metrics=result.metrics,
                    eval_mode=eval_mode,
                    model_cfg=model_cfg,
                ))

            self.completed_stages.append(stage_name)

        # --------------------------------------------------
        # PIPELINE FINISHED SUCCESSFULLY
        # --------------------------------------------------
        self.run_manager.write_pipeline_status(stage_names=self.topological_order)
        if table_entries:
            all_metrics["table_summary"] = table_entries
        return all_metrics
