"""Parse diagnostics of EVERY parser must be persisted per stage.

Covers audit finding A4 (docs/audit-2026-09-02.md): workflow stages pass
their own registry parser per step, so diagnostics live under cache_ids
other than the model's default parser. The executor used to save only
``stats.parse_diagnostics[model.parser.cache_id]`` — every step-parser
failure vanished silently.
"""

from stagekit.models.base import LLMBaseModel
from stagekit.pipeline.executor import PipelineExecutor
from stagekit.utils.parse_diagnostics import ParseDiagnostics


def _diag(parser_id: str, item_id: str) -> ParseDiagnostics:
    return ParseDiagnostics(
        parser_id=parser_id, backend="test", item_id=item_id,
        attempt=0, prompt_version="v1",
        raw_response="raw", parsed_output=None, error="parse failed",
    )


class _RecordingManager:
    def __init__(self):
        self.saved: dict[str, list] = {}

    def save_llm_stage_stats(self, stage_name, stats):
        pass

    def save_parse_diagnostics(self, stage_name, diags):
        self.saved[stage_name] = diags


class _Stats:
    retry_exhausted_items: dict = {}

    def __init__(self, buckets: dict):
        self.parse_diagnostics = buckets


class _ExecutorStub:
    """Just enough of PipelineExecutor for _save_llm_diagnostics."""

    def __init__(self, manager):
        self.run_manager = manager


def _model_with(buckets: dict) -> LLMBaseModel:
    model = object.__new__(LLMBaseModel)
    model.llm_stats = _Stats(buckets)
    model.parser = type("_P", (), {"cache_id": "default_parser"})()
    return model


class TestAllParserBucketsPersisted:
    def test_step_parser_diagnostics_are_not_dropped(self):
        model = _model_with({
            "default_parser": [_diag("default_parser", "a")],
            "step_parser_2": [_diag("step_parser_2", "b"),
                              _diag("step_parser_2", "c")],
        })
        mgr = _RecordingManager()

        PipelineExecutor._save_llm_diagnostics(_ExecutorStub(mgr), "stage", model)

        saved = mgr.saved["stage"]
        assert len(saved) == 3
        assert {d.parser_id for d in saved} == {"default_parser", "step_parser_2"}

    def test_workflow_stage_without_default_parser_entries(self):
        """The old code looked up model.parser and found nothing at all."""
        model = _model_with({"step_parser": [_diag("step_parser", "x")]})
        mgr = _RecordingManager()

        PipelineExecutor._save_llm_diagnostics(_ExecutorStub(mgr), "stage", model)

        assert [d.item_id for d in mgr.saved["stage"]] == ["x"]

    def test_no_diagnostics_saves_nothing(self):
        model = _model_with({})
        mgr = _RecordingManager()

        PipelineExecutor._save_llm_diagnostics(_ExecutorStub(mgr), "stage", model)

        assert mgr.saved == {}
