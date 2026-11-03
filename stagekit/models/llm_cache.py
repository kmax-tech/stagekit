# stagekit/models/llm_cache.py

import fcntl
import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from stagekit.utils.serialization import save_json

logger = logging.getLogger(__name__)

#: Slot-keyed sections of a cache entry. A slot is ``str(prompt_nbr)`` for
#: the run-global default, or ``"{prompt_nbr}:{sample_slot}"`` when a task
#: requests per-call sampling (the colon keeps both namespaces disjoint:
#: a bare prompt_nbr slot never contains one).
#: Single source of truth for every reader that walks entry structure —
#: scripts/merge_caches.py merges exactly these sections, so a section
#: added here without being slot-keyed dicts would break that contract.
ENTRY_SLOT_SECTIONS: tuple[str, ...] = ("answers", "raw", "thinking")


class CacheCorruptionError(RuntimeError):
    """A cache file exists but cannot be parsed as valid cache data.

    Raised instead of treating the file as empty: reading a corrupt shard
    as ``{"entries": {}}`` would let the next write REPLACE the file and
    silently destroy every entry it held. The remedy is manual — repair
    the file or delete it deliberately — so the run must stop here.
    """


@contextmanager
def _file_lock(path: Path) -> Generator[None, None, None]:
    """Cross-process exclusive lock for read-modify-write on *path*.

    Uses ``fcntl.flock`` on a SIDECAR file (``<name>.lock``): shard
    writes go through ``save_json`` (temp file + ``os.replace``), which
    swaps the target's inode — a lock on the data file itself would
    silently stop protecting it after the first write.  POSIX only
    (macOS/Linux).

    Side effects:
        Creates an empty ``<path>.lock`` next to the locked file
        (gitignored; safe to delete when no process is running).
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class LLMCache:
    """File-backed LLM response cache with per-shard locking.

    Design intent — result cache, not raw-data cache:
        The cache stores *parsed* results (``answers``) as the primary
        value.  Cache hits return these directly — no re-parsing.
        Raw LLM output (``raw``) is stored optionally alongside for
        diagnostics and auditing, but is never used for serving.

        This means parser changes do NOT retroactively alter cached
        results.  When parser semantics change, bump ``prompt_version``
        so old entries are bypassed.  If old raw data needs re-evaluation,
        write a dedicated migration script — do not change cache-hit
        behaviour.

    Use the ``get()`` factory instead of ``__init__()`` directly —
    it returns the same instance for the same ``cache_dir`` within a
    process, so all model instances sharing a cache directory share
    the same locks automatically.

    Thread safety:
        All shard reads/writes are protected by per-file threading locks.
        Multiple threads within the same process are safe.

    Process safety:
        Writes (``get_or_set``, ``_store_prompt``) additionally hold a
        cross-process ``flock`` on a sidecar ``<shard>.lock`` file, so
        concurrent *processes* sharing a ``cache_dir`` cannot lose each
        other's updates.  Reads take no file lock: a stale read is just
        a cache miss, and the subsequent write merges under the lock.
        ``save_json()`` writes atomically (temp + rename), so files are
        never corrupted even on crash.
    """

    # --------------------------------------------------
    # Singleton factory: one instance per cache_dir
    # --------------------------------------------------
    _instances: Dict[str, "LLMCache"] = {}
    _instances_guard = threading.Lock()

    _FACTORY_SENTINEL = object()

    @classmethod
    def get(cls, cache_dir: Path) -> "LLMCache":
        """Return the process-wide LLMCache for *cache_dir*.

        Creates a new instance on first call; returns the existing one
        on subsequent calls with the same resolved path.

        This is the only supported way to obtain an LLMCache.
        """
        key = str(cache_dir.resolve())
        with cls._instances_guard:
            if key not in cls._instances:
                cls._instances[key] = cls(cache_dir, _factory=cls._FACTORY_SENTINEL)
            return cls._instances[key]

    def __init__(self, cache_dir: Path, *, _factory: object = None):
        if _factory is not self._FACTORY_SENTINEL:
            logger.warning(
                "LLMCache() called directly — use LLMCache.get(cache_dir) "
                "to ensure shared locks across consumers."
            )
        self.cache_dir = cache_dir
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, path: str) -> threading.Lock:
        """Return a lock for the given file path."""
        with self._locks_guard:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]

    # --------------------------------------------------
    # Key computation
    # --------------------------------------------------
    def make_key(
        self,
        *,
        backend: str,
        prompt: str,
        model: str,
        temperature: Optional[float],
        parser_id: str,
        prompt_version: str | int = "v1",
        backend_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compute the cache key for one prompt.

        Args:
            backend: Backend name — two backends serving the same model id
                are two instruments and must not share entries.
            prompt: The rendered prompt text.
            model: Model id as configured.
            temperature: Sampling temperature, or None for a backend that
                does not consume it — then it is omitted from the payload,
                so a knob the backend never sends cannot split the cache
                (same rule as the repro hash, see base._build_cache_key).
            parser_id: Parser identity; a changed parser invalidates.
            prompt_version: Bumped when prompt semantics change.
            backend_params: Further output-affecting settings this backend
                consumes (e.g. max_tokens). Omitted from the payload when
                empty, so every key written before this parameter existed
                still resolves.

        Returns:
            The sha256 hex digest of the canonical payload.
        """
        payload = {
            "backend": backend,
            "prompt": prompt,
            "model": model,
            "parser": parser_id,
            "prompt_version": prompt_version,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if backend_params:
            payload["backend_params"] = backend_params
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # --------------------------------------------------
    # Path resolution
    # --------------------------------------------------
    def _path(
        self,
        *,
        key: str,
        dataset_id: str,
        model_id: str,
        parser_id: str,
        prompt_version: str | int,
    ) -> Path:
        shard = key[:2]
        return (
            self.cache_dir
            / dataset_id
            / model_id
            / parser_id
            / str(prompt_version)
            / f"{shard}.json"
        )

    def _prompts_path(
        self,
        *,
        dataset_id: str,
        model_id: str,
        parser_id: str,
        prompt_version: str | int,
    ) -> Path:
        return (
            self.cache_dir
            / dataset_id
            / model_id
            / parser_id
            / str(prompt_version)
            / "_prompts.json"
        )

    # --------------------------------------------------
    # Prompt storage
    # --------------------------------------------------
    def _store_prompt(
        self,
        prompt_hash: str,
        prompt_text: str,
        *,
        dataset_id: str,
        model_id: str,
        parser_id: str,
        prompt_version: str | int,
    ) -> None:
        """Store prompt text in _prompts.json, keyed by hash."""
        path = self._prompts_path(
            dataset_id=dataset_id,
            model_id=model_id,
            parser_id=parser_id,
            prompt_version=prompt_version,
        )

        lock = self._lock_for(str(path))
        with lock, _file_lock(path):
            if path.exists():
                try:
                    prompts = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    # Same rationale as _read_shard: resetting to {} would
                    # make the save below destroy every stored prompt text.
                    raise CacheCorruptionError(
                        f"Prompt store is not valid JSON: {path} — repair "
                        f"it or delete it deliberately, then re-run."
                    ) from e
            else:
                prompts = {}

            if prompt_hash in prompts:
                if prompts[prompt_hash] != prompt_text:
                    logger.warning(
                        "Prompt hash collision: hash %s has different text", prompt_hash[:16]
                    )
                return

            prompts[prompt_hash] = prompt_text
            save_json(prompts, path)

    # --------------------------------------------------
    # Shared shard I/O (under lock)
    # --------------------------------------------------
    def _read_shard(self, path: Path) -> dict:
        """Read shard file, returning {"entries": {}} only when it is absent.

        Raises:
            CacheCorruptionError: If the file exists but is not valid JSON
                or lacks an ``entries`` dict. Deliberately loud: the next
                ``get_or_set`` would otherwise overwrite the shard and
                silently destroy all of its entries (fail-loud,
                coding-style §1).
        """
        if not path.exists():
            return {"entries": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CacheCorruptionError(
                f"Cache shard is not valid JSON: {path} — repair it or "
                f"delete it deliberately, then re-run (line {e.lineno})."
            ) from e
        if not isinstance(data.get("entries"), dict):
            raise CacheCorruptionError(
                f"Cache shard has no 'entries' dict: {path} — repair it or "
                f"delete it deliberately, then re-run."
            )
        return data

    # --------------------------------------------------
    # Read API
    # --------------------------------------------------
    def get_one(
        self,
        *,
        key: str,
        slot: int | str,
        dataset_id: str,
        model_id: str,
        parser_id: str,
        prompt_version: str | int,
    ) -> Optional[Any]:
        """Return the answer stored under *slot*, or None.

        The lookup is an exact dict match on ``str(slot)`` — slot ``"1"``
        never resolves a sampled slot like ``"1:v1"`` and vice versa.
        """
        path = self._path(
            key=key, dataset_id=dataset_id, model_id=model_id,
            parser_id=parser_id, prompt_version=prompt_version,
        )
        lock = self._lock_for(str(path))
        with lock:
            data = self._read_shard(path)
            entry = data["entries"].get(key)
        if entry is None:
            return None
        return entry.get("answers", {}).get(str(slot))

    def get_all(
        self,
        *,
        key: str,
        dataset_id: str,
        model_id: str,
        parser_id: str,
        prompt_version: str | int,
    ) -> Dict[str, Any]:
        """Return all slots of an entry, keyed by slot string.

        Keys stay strings: sampled slots (``"1:v1"``) have no integer
        form, so an ``int()`` cast would crash on any entry they touch.
        """
        path = self._path(
            key=key, dataset_id=dataset_id, model_id=model_id,
            parser_id=parser_id, prompt_version=prompt_version,
        )
        lock = self._lock_for(str(path))
        with lock:
            data = self._read_shard(path)
            entry = data["entries"].get(key)
        if entry is None:
            return {}
        return dict(entry.get("answers", {}))

    def get_entry(
        self,
        *,
        key: str,
        dataset_id: str,
        model_id: str,
        parser_id: str,
        prompt_version: str | int,
    ) -> Optional[dict]:
        """Return full entry with prompt text resolved. For debugging/auditing.

        The one read path that tolerates a corrupt shard (warns, returns
        None): it is read-only and used for inspection, so it can never
        trigger the destructive overwrite that _read_shard's raise exists
        to prevent.
        """
        path = self._path(
            key=key, dataset_id=dataset_id, model_id=model_id,
            parser_id=parser_id, prompt_version=prompt_version,
        )
        lock = self._lock_for(str(path))
        with lock:
            try:
                data = self._read_shard(path)
            except CacheCorruptionError as e:
                logger.warning("get_entry: %s", e)
                return None
            entry = data["entries"].get(key)

        if entry is None:
            return None

        result = dict(entry)
        p_hash = entry.get("prompt_hash")
        if p_hash:
            prompts_path = self._prompts_path(
                dataset_id=dataset_id,
                model_id=model_id,
                parser_id=parser_id,
                prompt_version=prompt_version,
            )
            if prompts_path.exists():
                try:
                    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
                    result["prompt"] = prompts.get(p_hash)
                except json.JSONDecodeError:
                    pass  # debug-only prompt text; the cached result stands alone
        return result

    # --------------------------------------------------
    # Atomic get-or-set
    # --------------------------------------------------
    def get_or_set(
        self,
        *,
        key: str,
        slot: int | str,
        prompt: str,
        dataset_id: str,
        model_id: str,
        parser_id: str,
        prompt_version: str | int,
        value: Any,
        raw: Optional[str] = None,
        thinking: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, bool]:
        """Atomic read-or-write for a single cache slot.

        If the slot already exists, returns ``(existing_value, True)``.
        Otherwise writes *value* and returns ``(value, False)``.

        The entire check-and-write is under a single lock acquisition,
        eliminating the race window that existed with separate
        ``get_one()`` / ``set()`` calls.

        ``thinking`` is stored beside ``raw`` in its own sub-dict rather
        than nested inside it: ``raw[slot]`` is a plain string, and readers
        (scripts/merge_caches.py, the tests) rely on that.
        """
        path = self._path(
            key=key, dataset_id=dataset_id, model_id=model_id,
            parser_id=parser_id, prompt_version=prompt_version,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        p_hash = self._prompt_hash(prompt)
        slot = str(slot)

        # Thread lock (in-process) + flock (cross-process): the shard is
        # re-read inside both locks so concurrent writers merge instead
        # of overwriting each other's entries.
        lock = self._lock_for(str(path))
        with lock, _file_lock(path):
            data = self._read_shard(path)
            entries = data.setdefault("entries", {})
            entry = entries.setdefault(
                key,
                {"prompt_hash": p_hash, "answers": {}, "raw": {}, "meta": {}},
            )

            entry["prompt_hash"] = p_hash
            answers = entry.setdefault("answers", {})

            if slot in answers:
                return answers[slot], True

            answers[slot] = value
            if raw is not None:
                entry.setdefault("raw", {})[slot] = raw
            if thinking:
                entry.setdefault("thinking", {})[slot] = thinking
            if meta:
                entry.setdefault("meta", {}).update(meta)

            save_json(data, path)

        # Store prompt text outside shard lock
        self._store_prompt(
            p_hash, prompt,
            dataset_id=dataset_id, model_id=model_id,
            parser_id=parser_id, prompt_version=prompt_version,
        )

        return value, False

