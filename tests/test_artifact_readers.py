"""Tests for the typed artifact readers (validate-at-read + typed access)."""

import pytest

from stagekit.stages.artifact_readers import Consensus, Judgement, PerRecord, Scores, ScoresCV
from stagekit.stages.artifact_schemas import ArtifactSchemaError


def _scores_dict(**overrides):
    d = {
        "accuracy": 0.8, "precision": 0.75, "recall": 0.9,
        "f1": 0.82, "f1_macro": 0.79, "n_examples": 2,
        "per_record": [
            {"id": "a", "gold": 1, "pred": 1},
            {"id": "b", "gold": 0, "pred": 1},
        ],
    }
    d.update(overrides)
    return d


class TestScoresReader:

    def test_roundtrip_typed_access(self):
        scores = Scores.from_artifact(_scores_dict())
        assert scores.f1_macro == 0.79
        assert scores.n_examples == 2
        assert scores.per_record[0] == PerRecord(id="a", gold=1, pred=1)

    def test_missing_f1_macro_raises_with_source(self):
        d = _scores_dict()
        del d["f1_macro"]
        with pytest.raises(ArtifactSchemaError, match="f1_macro"):
            Scores.from_artifact(d, source="output/run_x")

    def test_non_dict_raises(self):
        with pytest.raises(ArtifactSchemaError, match="expected dict"):
            Scores.from_artifact([1, 2, 3])  # type: ignore[arg-type]

    def test_frozen(self):
        scores = Scores.from_artifact(_scores_dict())
        with pytest.raises(Exception):
            scores.f1 = 0.0  # type: ignore[misc]


class TestScoresCVReader:

    def test_mean_std_pairs(self):
        d = {
            "n_examples": 10, "n_folds": 2,
            "accuracy": 0.8, "accuracy_std": 0.05,
            "precision": 0.7, "precision_std": 0.02,
            "recall": 0.85, "recall_std": 0.03,
            "f1": 0.79, "f1_std": 0.04,
            "f1_macro": 0.77, "f1_macro_std": 0.04,
            "per_fold": [{"fold": 0}, {"fold": 1}],
        }
        cv = ScoresCV.from_artifact(d)
        assert cv.f1_macro.mean == 0.77
        assert cv.f1_macro.std == 0.04
        assert cv.n_folds == 2


class TestJudgementReader:

    def test_typed_access_and_overall(self):
        j = Judgement.from_artifact({
            "judged_prompt": "binary_zs", "prompt_version": "abc",
            "judge_model": "gpt", "sample_ids": ["r1"],
            "scores": {"clarity": 4, "guideline_coverage": 3, "overall": 3.5},
            "weaknesses": ["w"], "suggested_rules": ["r"],
        })
        assert j.overall == 3.5
        assert j.weaknesses == ("w",)

    def test_scores_without_overall_raise(self):
        with pytest.raises(ArtifactSchemaError, match="overall"):
            Judgement.from_artifact({
                "judged_prompt": "p", "prompt_version": "v",
                "judge_model": "m", "sample_ids": [],
                "scores": {"clarity": 4}, "weaknesses": [], "suggested_rules": [],
            })


class TestConsensusReader:

    def test_typed_mean_std(self):
        c = Consensus.from_artifact({
            "judges": ["gpt", "llama"], "judged_prompt": "p",
            "prompt_version": "v",
            "scores": {"overall": {"mean": 3.5, "std": 0.5}},
            "shared_weaknesses": ["w"], "shared_suggested_rules": [],
        })
        assert c.scores["overall"].mean == 3.5
        assert c.judges == ("gpt", "llama")
