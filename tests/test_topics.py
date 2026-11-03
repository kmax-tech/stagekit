"""Loading topics: each refusal names the topic, a loaded topic adds its names."""

import textwrap

import pytest

from stagekit.run import topics
from stagekit.utils.registry import LazyRegistry

REG = LazyRegistry("thing", {"core_thing": "stagekit.utils.registry"})


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    """One registry of kind "thing", no topic loaded, topic modules from tmp_path."""
    fresh = LazyRegistry("thing", {"core_thing": "stagekit.utils.registry"})
    monkeypatch.setattr(topics, "REGISTRY_HOMES", {"thing": "tests.test_topics:REG"})
    monkeypatch.setattr(topics, "_loaded", [])
    monkeypatch.setattr(f"{__name__}.REG", fresh)
    monkeypatch.syspath_prepend(str(tmp_path))

    def write_topic(name: str, body: str) -> str:
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
        return name

    return fresh, write_topic


def test_a_loaded_topic_adds_its_names(isolated):
    reg, write_topic = isolated
    topic = write_topic("topic_ok", """
        REGISTRATIONS = {"thing": {"a": "topic_ok_items"}}
    """)
    topics.load_topics([topic])
    topics.load_topics([topic])  # idempotent
    assert reg.available() == ["a", "core_thing"]
    assert topics.loaded_topics() == (topic,)


def test_an_unimportable_topic_is_named(isolated):
    with pytest.raises(topics.TopicError, match="topic 'no_such_topic' .* cannot be imported"):
        topics.load_topics(["no_such_topic"])


def test_a_topic_without_registrations_is_named(isolated):
    _, write_topic = isolated
    topic = write_topic("topic_bare", "X = 1\n")
    with pytest.raises(topics.TopicError, match="topic 'topic_bare' has no REGISTRATIONS"):
        topics.load_topics([topic])


def test_a_kind_no_registry_knows_is_named_with_its_topic(isolated):
    _, write_topic = isolated
    topic = write_topic("topic_kind", """
        REGISTRATIONS = {"widget": {"a": "m"}}
    """)
    with pytest.raises(topics.TopicError, match=r"topic 'topic_kind' registers under \['widget'\]"):
        topics.load_topics([topic])


def test_a_name_mapped_elsewhere_is_refused_without_half_loading(isolated):
    reg, write_topic = isolated
    topic = write_topic("topic_clash", """
        REGISTRATIONS = {"thing": {"core_thing": "elsewhere"}}
    """)
    with pytest.raises(topics.TopicError, match="topic 'topic_clash' .*core_thing"):
        topics.load_topics([topic])
    assert reg.available() == ["core_thing"]
    assert topics.loaded_topics() == ()


def test_an_unknown_name_says_which_topics_were_loaded(isolated):
    reg, write_topic = isolated
    topics.load_topics([write_topic("topic_named", 'REGISTRATIONS = {"thing": {"a": "m"}}\n')])
    with pytest.raises(ValueError, match=r"Unknown thing: 'b' \(topics that added thing names: \['topic_named'\]\)"):
        reg.get("b")


def test_every_framework_registry_has_its_home():
    """REGISTRY_HOMES resolves; the probe in test_core_boundary.py checks
    that no LazyRegistry is missing from it."""
    for kind in topics.REGISTRY_HOMES:
        assert topics.registry_of(kind)._kind == kind
