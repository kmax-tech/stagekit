# stagekit/utils/parse_diagnostics.py

from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ParseDiagnostics:
    parser_id: str
    backend: str
    item_id: Optional[str]
    attempt: int
    prompt_version: str

    raw_response: str
    parsed_output: Optional[Any]
    error: Optional[str] = None

    #: The reasoning trace, kept apart from raw_response. Empty for models
    #: that do not think, and for every cache hit (see LLMBaseModel).
    thinking: str = ""
