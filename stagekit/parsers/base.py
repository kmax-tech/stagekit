import json
import logging

import json_repair
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_json_lenient(
    response_str: str,
    extract: "Callable[[Any], Optional[dict]]",
    *,
    parser_name: str,
) -> Optional[dict]:
    """Lenient JSON scan over an LLM response — a DELIBERATE tolerance.

    LLM output often wraps its JSON payload in prose or code fences, so
    strict parsing would reject valid answers.  Strategy: try the full
    text as JSON first, then scan lines bottom-up for the last line
    starting with ``{``.

    Args:
        response_str: Raw LLM response text.
        extract: Maps a decoded JSON candidate to a result dict, or
            returns None when the candidate does not carry the expected
            payload.  ``AttributeError`` from non-dict candidates is
            treated as a non-match.
        parser_name: Label for the failure warning.

    Returns:
        The first extracted result, or None when nothing matches.
        A None return is not silent: this logs a warning, and the
        caller's retry/diagnostics layer (``LLMCallResult``, parse
        diagnostics) counts the failure.
    """
    text = response_str.strip()
    if not text:
        logger.warning("%s: empty response", parser_name)
        return None

    def _try(candidate: str) -> Optional[dict]:
        try:
            obj = json.loads(candidate)
            return extract(obj)
        except (json.JSONDecodeError, AttributeError):
            return None

    # Fast path: full string is JSON
    result = _try(text)
    if result is not None:
        return result

    # Fallback: last JSON-looking line in a multi-line response
    for line in reversed(text.splitlines()):
        line = line.strip().strip("`")
        if line.startswith("{"):
            result = _try(line)
            if result is not None:
                return result

    logger.warning("%s: no valid JSON with expected payload found", parser_name)
    return None


class BaseResponseParser:
    """
    Flexible base class for all parsers.

    Responsibilities:
    - extract_fields_from_record(): pull prompt fields from a record dict
    - prepare_fields(): modify fields before template rendering
    - parse(): convert LLM output to structured result
    """

    # Identifier used as part of the cache key.  The cache key includes
    # (backend, prompt, model, temperature, parser_id, prompt_version).
    # prompt_version is auto-derived from the template content hash, but
    # cache_id is manually set per parser subclass.
    #
    # CONTRACT: bump cache_id whenever parse logic changes in a way that
    # would produce different output for the same raw LLM response.
    # If cache_id is not updated, stale cached entries will be served
    # without re-parsing — the cache has no way to detect logic changes.
    cache_id: str = "base"

    # Required JSON keys in the LLM output (e.g. ["labels"]).
    # Subclasses that use it for validation (e.g. BayesianProbabilityParser)
    # reference self.OUTPUT_REQUIRED in parse().
    OUTPUT_REQUIRED: List[str] = []

    # Optional JSON Schema for schema-guided repair via json_repair.
    # When set, repair_json uses it to coerce types and fill defaults.
    OUTPUT_SCHEMA: Optional[Dict[str, Any]] = None

    # ============================================================
    # Record → Fields
    # ============================================================
    def extract_fields_from_record(self, record: dict) -> dict:
        """
        Extract prompt fields from a record dict.

        Should return a dict of fields to be used inside the
        prompt template, e.g.:

        return {
            "claim": ...,
            "evidence": ...,
            "allowed_labels": ...
        }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement extract_fields_from_record()"
        )

    # ============================================================
    # Field Pre-processing (before template rendering)
    # ============================================================
    def prepare_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allow the parser to clean/modify/inject values.
        Example: lowercase labels, validate fields, add defaults, etc.
        """
        return fields

    # ============================================================
    # JSON Repair
    # ============================================================
    @classmethod
    def repair_json(cls, response_str: str) -> str:
        """Repair malformed JSON from LLM output using json_repair.

        Handles: missing/extra quotes, unescaped characters, control characters,
        markdown fences, truncated output, misplaced commas, and more.

        When OUTPUT_SCHEMA is defined on the parser, uses schema-guided repair
        to coerce types, fill missing required fields, and drop disallowed keys.
        If schema-guided repair fails (e.g. type coercion impossible), falls
        back to schema-less repair — type enforcement belongs in parse().

        Returns a valid JSON string ready for json.loads().
        """
        kwargs: dict = {"ensure_ascii": False}
        if cls.OUTPUT_SCHEMA is not None:
            kwargs["schema"] = cls.OUTPUT_SCHEMA
        try:
            repaired = json_repair.repair_json(response_str, **kwargs)
        except (ValueError, TypeError, KeyError):
            # Schema-guided repair can fail on type mismatches (e.g. null
            # where string expected, dict in string[] array).  Fall back to
            # schema-less repair so parse() can handle filtering.
            if cls.OUTPUT_SCHEMA is not None:
                logger.debug(
                    "[%s] Schema-guided repair failed, falling back to schema-less",
                    cls.__name__,
                )
                repaired = json_repair.repair_json(
                    response_str, ensure_ascii=False,
                )
            else:
                raise
        if isinstance(repaired, str):
            return repaired
        if isinstance(repaired, tuple):
            repaired = repaired[0]
        return json.dumps(repaired, ensure_ascii=False)

    # ============================================================
    # Output Parsing
    # ============================================================
    def parse(self, response_str: str, fields: Optional[Dict[str, Any]] = None):
        """
        Convert raw LLM response into structured Python data.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    # ============================================================
    # Safe Parse Wrapper (with failure logging)
    # ============================================================
    def parse_with_logging(
        self,
        response_str: str,
        *,
        fields: Optional[Dict[str, Any]] = None,
        entry_meta: Optional[Dict[str, Any]] = None,
        template_str: Optional[str] = None,
    ) -> Optional[Any]:
        """Repair, parse, gate (OUTPUT_REQUIRED, template echo), log failures.

        Returns:
            The parsed result, or None on any rejection. (This used to
            return a ``(result, used_fallback)`` tuple; the flag was
            never set and reported a fallback mechanism that does not
            exist — audit C3.)
        """
        response_str = self.repair_json(response_str)

        try:
            result = self.parse(response_str, fields)
        except Exception as e:
            logger.error("[%s] Parse exception: %s", self.__class__.__name__, e)
            self._log_failed_entry(response_str, fields, entry_meta)
            raise

        # Enforce OUTPUT_REQUIRED — reject results missing declared keys
        if result is not None and self.OUTPUT_REQUIRED:
            if not isinstance(result, dict):
                logger.warning(
                    "[%s] OUTPUT_REQUIRED declared but parse() returned %s, not dict",
                    self.__class__.__name__, type(result).__name__,
                )
                result = None
            else:
                missing = [k for k in self.OUTPUT_REQUIRED if k not in result]
                if missing:
                    logger.warning(
                        "[%s] parse() result missing OUTPUT_REQUIRED keys: %s",
                        self.__class__.__name__, missing,
                    )
                    self._log_failed_entry(response_str, fields, entry_meta)
                    result = None

        if result is not None and template_str is not None:
            if self._is_template_example(result, template_str):
                logger.error("[%s] Parse returned template example verbatim", self.__class__.__name__)
                self._log_failed_entry(response_str, fields, entry_meta)
                result = None

        if result is None:
            logger.debug("[%s] Parse returned None", self.__class__.__name__)
            self._log_failed_entry(response_str, fields, entry_meta)

        return result

    def _is_template_example(self, result: dict, template_str: str) -> bool:
        """Return True if result matches the example JSON embedded in the template."""
        try:
            repaired = json_repair.repair_json(template_str, ensure_ascii=False)
            example = json.loads(repaired) if isinstance(repaired, str) else repaired
            if not isinstance(example, dict):
                return False
            for key, template_val in example.items():
                result_val = result.get(key)
                if result_val is None:
                    continue
                if isinstance(template_val, list) and isinstance(result_val, list):
                    if sorted(str(v) for v in result_val) == sorted(str(v) for v in template_val):
                        return True
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug("Failed to parse template example: %s", exc)
        return False

    # ============================================================
    # Failure Logger
    # ============================================================
    def _log_failed_entry(
        self,
        response_str: str,
        fields: Optional[Dict[str, Any]],
        entry_meta: Optional[Dict[str, Any]],
    ):
        parts = [
            f"PARSER FAILURE — {self.__class__.__name__} (cache_id={self.cache_id})",
        ]

        if entry_meta:
            meta_str = ", ".join(f"{k}={v}" for k, v in entry_meta.items())
            parts.append(f"Meta: {meta_str}")

        if fields:
            try:
                parts.append(f"Fields: {json.dumps(fields, ensure_ascii=False)}")
            except (TypeError, ValueError):
                parts.append(f"Fields: {fields}")

        parts.append(f"Raw output: {response_str}")

        logger.debug("\n".join(parts))
