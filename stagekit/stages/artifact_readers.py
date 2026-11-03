"""Typed readers for cross-run artifact forms.

Artifacts travel as JSON dicts (they cross run and process boundaries),
so their FORM is enforced by the declarative schemas in
artifact_schemas.py at emit and preload time. These readers add the
third leg: a CONSUMER that loads an artifact gets schema validation at
the read boundary plus typed attribute access afterwards — so
downstream code (refinement driver, notebooks, app) is pyright-checked
instead of stringly-typed ``scores["f1_macro"]`` lookups.

Pattern: ``Scores.from_artifact(raw, source="run dir …")`` validates
against the same schema the producer was held to, then freezes the
values into an immutable dataclass. Producers keep emitting plain
dicts — the schema stays the single contract; readers only consume it.

The k-fold split artifact deliberately has NO reader here: it already
has a typed contract (``KFoldAssignment`` TypedDict +
``validate_kfold_assignment`` in split_config.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from stagekit.stages.artifact_schemas import (
    ArtifactSchemaError,
    CONSENSUS,
    JUDGEMENT,
    PREDICTION_SCORES,
    PREDICTION_SCORES_CV,
    validate_artifact,
)


def _validated(value: dict, schema: dict, *, reader: str, source: str) -> dict:
    """Validate *value* against *schema*, raising with consumer context."""
    where = f"{reader}({source})" if source else reader
    if not isinstance(value, dict):
        raise ArtifactSchemaError(
            where, [f"expected dict artifact, got {type(value).__name__}"],
        )
    errors = validate_artifact(value, schema, stage_name=reader, artifact_key=source or "artifact")
    if errors:
        raise ArtifactSchemaError(where, errors)
    return value


@dataclass(frozen=True)
class PerRecord:
    """One scored prediction: gold vs pred for a record id."""
    id: str | int
    gold: int
    pred: int


@dataclass(frozen=True)
class Scores:
    """Single-run scores artifact (eval_predictions, eval_mode=single_run)."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    f1_macro: float
    n_examples: int
    per_record: tuple[PerRecord, ...]

    @classmethod
    def from_artifact(cls, value: dict, *, source: str = "") -> "Scores":
        v = _validated(value, PREDICTION_SCORES, reader="Scores", source=source)
        return cls(
            accuracy=float(v["accuracy"]),
            precision=float(v["precision"]),
            recall=float(v["recall"]),
            f1=float(v["f1"]),
            f1_macro=float(v["f1_macro"]),
            n_examples=int(v["n_examples"]),
            per_record=tuple(
                PerRecord(id=r["id"], gold=int(r["gold"]), pred=int(r["pred"]))
                for r in v["per_record"]
            ),
        )


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float


@dataclass(frozen=True)
class ScoresCV:
    """K-fold CV scores artifact (eval_predictions with split_artifact)."""
    n_examples: int
    n_folds: int
    accuracy: MeanStd
    precision: MeanStd
    recall: MeanStd
    f1: MeanStd
    f1_macro: MeanStd
    per_fold: tuple[dict, ...]  # fold dicts incl. their own per_record

    @classmethod
    def from_artifact(cls, value: dict, *, source: str = "") -> "ScoresCV":
        v = _validated(value, PREDICTION_SCORES_CV, reader="ScoresCV", source=source)

        def ms(name: str) -> MeanStd:
            return MeanStd(mean=float(v[name]), std=float(v[f"{name}_std"]))

        return cls(
            n_examples=int(v["n_examples"]),
            n_folds=int(v["n_folds"]),
            accuracy=ms("accuracy"),
            precision=ms("precision"),
            recall=ms("recall"),
            f1=ms("f1"),
            f1_macro=ms("f1_macro"),
            per_fold=tuple(v["per_fold"]),
        )


@dataclass(frozen=True)
class Judgement:
    """One prompt-judge verdict (prompt_judge stage)."""
    judged_prompt: str
    prompt_version: str
    judge_model: str
    scores: dict[str, float]
    weaknesses: tuple[str, ...]
    suggested_rules: tuple[str, ...]
    sample_ids: tuple

    @property
    def overall(self) -> float:
        return float(self.scores["overall"])

    @classmethod
    def from_artifact(cls, value: dict, *, source: str = "") -> "Judgement":
        v = _validated(value, JUDGEMENT, reader="Judgement", source=source)
        return cls(
            judged_prompt=v["judged_prompt"],
            prompt_version=v["prompt_version"],
            judge_model=v["judge_model"],
            scores={k: float(s) for k, s in v["scores"].items()},
            weaknesses=tuple(v["weaknesses"]),
            suggested_rules=tuple(v["suggested_rules"]),
            sample_ids=tuple(v["sample_ids"]),
        )


@dataclass(frozen=True)
class Consensus:
    """Multi-judge consensus (judge_consensus stage)."""
    judges: tuple[str, ...]
    judged_prompt: str
    prompt_version: str
    scores: dict[str, MeanStd]
    shared_weaknesses: tuple[str, ...]
    shared_suggested_rules: tuple[str, ...]

    @classmethod
    def from_artifact(cls, value: dict, *, source: str = "") -> "Consensus":
        v = _validated(value, CONSENSUS, reader="Consensus", source=source)
        return cls(
            judges=tuple(v["judges"]),
            judged_prompt=v["judged_prompt"],
            prompt_version=v["prompt_version"],
            scores={
                dim: MeanStd(mean=float(s["mean"]), std=float(s["std"]))
                for dim, s in v["scores"].items()
            },
            shared_weaknesses=tuple(v["shared_weaknesses"]),
            shared_suggested_rules=tuple(v["shared_suggested_rules"]),
        )
