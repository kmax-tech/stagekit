"""Prompts registered from a directory: each file under its name, refusals loud."""

import pytest

from stagekit.prompts import registry
from stagekit.prompts.base import PromptRenderer
from stagekit.utils.registry import LazyRegistry


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    reg = LazyRegistry("prompt", {})
    monkeypatch.setattr(registry, "PROMPT_REGISTRY", reg)
    return reg


def test_each_file_registers_under_its_name(tmp_path, fresh_registry):
    (tmp_path / "judge_pair.txt").write_text("Compare $left with $right. $left first.")
    (tmp_path / "plain.txt").write_text("No fields.")
    (tmp_path / ".DS_Store").write_bytes(b"\0")

    assert registry.register_prompt_dir(tmp_path) == ["judge_pair", "plain"]
    template = fresh_registry.get("judge_pair")
    assert template.required_fields == ["left", "right"]
    assert PromptRenderer.render(template, {"left": "A", "right": "B"}) == "Compare A with B. A first."
    assert fresh_registry.get("plain").required_fields == []


def test_a_file_of_another_kind_is_refused(tmp_path, fresh_registry):
    (tmp_path / "ok.txt").write_text("x")
    (tmp_path / "notes.md").write_text("x")
    with pytest.raises(ValueError, match=r"more than <name>.txt files: \['notes.md'\]"):
        registry.register_prompt_dir(tmp_path)
    assert fresh_registry.available() == []


def test_a_taken_name_is_refused_and_nothing_registers(tmp_path, fresh_registry):
    fresh_registry.register("taken")(object())
    (tmp_path / "taken.txt").write_text("x")
    (tmp_path / "new.txt").write_text("x")
    with pytest.raises(ValueError, match=r"already registered: \['taken'\]"):
        registry.register_prompt_dir(tmp_path)
    assert fresh_registry.available() == ["taken"]


def test_an_empty_or_missing_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds no <name>.txt file"):
        registry.register_prompt_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.register_prompt_dir(tmp_path / "missing")
