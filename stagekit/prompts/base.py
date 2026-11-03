from dataclasses import dataclass
from typing import List, Dict, Any
from string import Template
import hashlib

@dataclass
class PromptTemplate:
    name: str
    template_str: str
    required_fields: List[str]


    @property
    def version(self) -> str:
        return hashlib.sha256(self.template_str.encode()).hexdigest()[:8]


class PromptRenderer:
    @staticmethod
    def render(template: PromptTemplate, fields: Dict[str, Any]) -> str:

        # validate required fields
        missing = [
            key for key in template.required_fields
            if key not in fields or fields[key] is None
        ]

        if missing:
            raise ValueError(
                f"Missing required fields for template: {missing}\n"
                f"Provided fields: {list(fields.keys())}"
            )

        # use Template instead of .format()
        t = Template(template.template_str)

        try:
            return t.substitute(fields)
        except KeyError as e:
            raise ValueError(
                f"Template placeholder missing in fields: {e}\n"
                f"Provided fields: {list(fields.keys())}"
            )
