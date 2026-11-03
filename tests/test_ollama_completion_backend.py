"""The raw-completion ollama backend: what one call sends, what it records,
and that it is a SEPARATE instrument from `ollama` in cache key and hash.

The identity tests go through the real config loader (`load_model_config`,
the strict OmegaConf merge) rather than a hand-built dict, because `stop`
is the first list-valued identity field and a ListConfig would break
`json.dumps` in the cache key and the repro hash if the loader let one
through.
"""

from dataclasses import asdict

import pytest

import stagekit.models.backends.ollama_backend as ollama_module
import stagekit.models.backends.ollama_completion_backend as completion_module
from stagekit.analysis.repro_hash import _model_canon, repro_hash_from_dict
from stagekit.config.io import load_model_config
from stagekit.models.backends.factory import BACKEND_CAPABILITIES, backend_consumes
from stagekit.models.backends.ollama_completion_backend import (
    UNAVAILABLE,
    OllamaCompletionBackend,
)
from tests.llm_harness import HarnessModel, meta_cfg, model_cfg

URL = "http://node:11434"
URL2 = "http://other:11434"


class _Listed:
    def __init__(self, model, digest):
        self.model = model
        self.digest = digest


class _FakeClient:
    """Stands in for ollama.Client; records the last generate() kwargs."""

    last_kwargs: dict = {}
    digests: dict = {}   # host -> digest listed for model "m"

    def __init__(self, response, host=None, timeout=None):
        self._response = response
        self.host = host
        self.timeout = timeout

    def list(self):
        digest = _FakeClient.digests.get(self.host, "a" * 64)
        return {"models": [_Listed("m", digest)]}

    def generate(self, **kwargs):
        _FakeClient.last_kwargs = kwargs
        return self._response


class _Resp:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise completion_module.requests.RequestException("down")

    def json(self):
        return self._payload


@pytest.fixture
def fake_servers(monkeypatch):
    """Patch the ollama client and the version probe; return a builder."""
    _FakeClient.digests = {}

    def make(response=None, *, digests=None, version="0.11.4", version_ok=True, **kwargs):
        if digests:
            _FakeClient.digests = digests
        monkeypatch.setattr(
            ollama_module.ollama, "Client",
            lambda host=None, timeout=None: _FakeClient(response or {"response": "x"}, host, timeout),
        )
        monkeypatch.setattr(
            completion_module.requests, "get",
            lambda url, timeout=None: _Resp({"version": version}, ok=version_ok),
        )
        kwargs.setdefault("server_urls", [URL])
        return OllamaCompletionBackend(model_name="m", temperature=0.0, **kwargs)
    return make


class TestPayload:

    def test_raw_and_all_knobs_are_sent(self, fake_servers):
        backend = fake_servers(max_tokens=512, stop=["###"], num_ctx=4096)
        backend.generate("### Instruction:\nx\n\n### Response:")

        kw = _FakeClient.last_kwargs
        assert kw["raw"] is True
        assert kw["prompt"] == "### Instruction:\nx\n\n### Response:"
        assert kw["options"] == {
            "temperature": 0.0, "num_predict": 512, "stop": ["###"], "num_ctx": 4096,
        }

    def test_unset_knobs_stay_out_of_the_payload(self, fake_servers):
        """None means 'server default' and must not be sent as a value."""
        fake_servers().generate("p")

        assert _FakeClient.last_kwargs["options"] == {"temperature": 0.0}

    def test_stop_is_a_plain_list_in_the_payload(self, fake_servers):
        backend = fake_servers(stop=("###",))
        backend.generate("p")

        assert _FakeClient.last_kwargs["options"]["stop"] == ["###"]
        assert type(_FakeClient.last_kwargs["options"]["stop"]) is list

    def test_answer_shape(self, fake_servers):
        out = fake_servers({"response": "the answer", "thinking": None}).generate("p")

        assert out.raw == "the answer"
        assert out.thinking == ""

    def test_empty_response_is_empty_string(self, fake_servers):
        """A stop sequence hit at once yields '' — a value, not a None."""
        out = fake_servers({"response": ""}).generate("p")

        assert out.raw == ""


class TestInstrumentIdentity:

    def test_digest_and_version_are_recorded(self, fake_servers):
        backend = fake_servers(digests={URL: "abc123def456" + "0" * 52}, version="0.11.4")

        assert backend.model_digest == "abc123def456"
        assert backend.server_version == "0.11.4"

    def test_servers_disagreeing_on_digest_refuse_to_start(self, fake_servers):
        with pytest.raises(RuntimeError, match="disagree"):
            fake_servers(
                digests={URL: "a" * 64, URL2: "b" * 64}, server_urls=[URL, URL2],
            )

    def test_missing_digest_fails_loud(self, fake_servers):
        with pytest.raises(RuntimeError, match="no digest"):
            fake_servers(digests={URL: ""})

    def test_version_probe_failure_is_recorded_not_fatal(self, fake_servers):
        backend = fake_servers(version_ok=False)

        assert backend.server_version == UNAVAILABLE

    def test_run_metadata_carries_digest_and_server_version(self, fake_servers):
        """RunManager.record_stage_execution reads both off the backend
        (same harness as tests/test_run_index.py::TestModelUsageDigest)."""
        import threading

        from stagekit.run.manager import RunManager

        backend = fake_servers(digests={URL: "abc123def456" + "0" * 52})

        class _Model:
            pass
        m = _Model()
        m.backend = backend  # type: ignore[attr-defined]  # duck-typed model

        class _Usage(RunManager):
            def __init__(self):  # no run directories for a pure recording test
                self._model_usage = {"stages": {}}
                self._usage_lock = threading.Lock()

        mgr = _Usage()
        mgr.record_stage_execution(
            stage_name="s", model_cfg=model_cfg(backend="ollama_completion"), model=m,
        )
        entry = mgr._model_usage["stages"]["s"]["model"]
        assert entry["digest"] == "abc123def456"
        assert entry["server_version"] == "0.11.4"


class TestIdentityDeclaration:

    def test_capabilities_declared(self):
        assert BACKEND_CAPABILITIES["ollama_completion"]["identity_fields"] == {
            "temperature", "max_tokens", "stop", "num_ctx",
        }
        assert OllamaCompletionBackend.config_fields == ("max_tokens", "stop", "num_ctx")

    def test_legacy_ollama_is_untouched(self):
        """The whole reason for a second backend: old keys and hashes stay valid."""
        for field in ("max_tokens", "stop", "num_ctx"):
            assert not backend_consumes("ollama", field)
        params = HarnessModel(
            model_cfg(backend="ollama", max_tokens=512, stop=["###"], num_ctx=4096)
        )._backend_params("ollama")
        assert params == {}

    def test_backend_params_carry_the_three_knobs(self):
        params = HarnessModel(
            model_cfg(backend="ollama_completion", max_tokens=512, stop=["###"], num_ctx=4096)
        )._backend_params("ollama_completion")
        assert params == {"max_tokens": 512, "stop": ["###"], "num_ctx": 4096}

    def test_same_model_two_ollama_backends_two_cache_keys(self, tmp_path):
        keys = []
        for backend in ("ollama", "ollama_completion"):
            model = HarnessModel(
                model_cfg(backend=backend, model_name="llama2:7b-chat", max_tokens=512),
                meta_cfg(enable_cache=True),
                cache_dir=tmp_path / backend,
            )
            key, _ = model._build_cache_key(backend, "the prompt", "binary", "v1", "ds")
            keys.append(key)
        assert keys[0] != keys[1]

    def test_model_canon_keeps_stop_only_where_consumed(self):
        cfg = asdict(model_cfg(backend="ollama_completion", stop=["###"], num_ctx=4096))
        assert _model_canon(cfg)["stop"] == ["###"]
        assert _model_canon(cfg)["num_ctx"] == 4096

        cfg = asdict(model_cfg(backend="ollama", stop=["###"], num_ctx=4096))
        assert "stop" not in _model_canon(cfg)
        assert "num_ctx" not in _model_canon(cfg)


#: A raw-completion model config as a project ships it (in fallacy-detection:
#: config/model/llama2-7b-chat-completion-model.yaml, checked there too).
COMPLETION_MODEL_YAML = """\
model_type: llm
backend: ollama_completion
model_name: llama2:7b-chat
temperature: 0.0
max_tokens: 512
stop: ["###"]
num_ctx: 4096
"""


@pytest.fixture()
def shipped_model():
    """The YAML in the test project's config/model/ (tests/conftest.py)."""
    from stagekit.config import paths

    path = paths.CONFIG_DIR / "model" / "completion-fixture-model.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(COMPLETION_MODEL_YAML)
    return path.stem


class TestRealConfigLoadPath:
    """A project's YAML through the strict loader: list stays a list."""

    def test_yaml_loads_with_plain_python_types(self, shipped_model):
        cfg = load_model_config(shipped_model)

        assert cfg.backend == "ollama_completion"
        assert cfg.model_name == "llama2:7b-chat"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512
        assert cfg.stop == ["###"] and type(cfg.stop) is list
        assert cfg.num_ctx == 4096

    def test_loaded_config_hashes_and_keys(self, tmp_path, shipped_model):
        """json.dumps in cache key and repro hash must accept the loaded values."""
        cfg = load_model_config(shipped_model)

        digest = repro_hash_from_dict({"model": asdict(cfg), "pipeline": {"stages": []}})
        assert digest.startswith("v") and len(digest.split("-", 1)[1]) == 64

        model = HarnessModel(cfg, meta_cfg(enable_cache=True), cache_dir=tmp_path)
        key, _ = model._build_cache_key("ollama_completion", "p", "binary", "v1", "ds")
        assert key and len(key) == 64

    def test_stop_changes_the_cache_key(self, tmp_path):
        """A list-valued identity field must actually reach the key."""
        keys = []
        for stop in (["###"], ["###", "\n\n"]):
            model = HarnessModel(
                model_cfg(backend="ollama_completion", stop=stop),
                meta_cfg(enable_cache=True),
                cache_dir=tmp_path / str(len(stop)),
            )
            key, _ = model._build_cache_key("ollama_completion", "p", "binary", "v1", "ds")
            keys.append(key)
        assert keys[0] != keys[1]
