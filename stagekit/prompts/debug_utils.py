import logging
from pathlib import Path

from stagekit.prompts.registry import get_prompt
from stagekit.prompts.base import PromptRenderer, PromptTemplate

logger = logging.getLogger(__name__)


def save_prompt_debug(
    *,
    llm_task_cfg,
    parser,
    record: dict,
    output_dir: Path,
):
    """
    Save prompt template and one rendered example for debugging.

    Used by llm_enrich which has an llm_task_cfg + parser pair.
    This utility is intentionally side-effect-only (writes files).
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    template = get_prompt(llm_task_cfg.template)

    (output_dir / "template_used.txt").write_text(
        template.template_str, encoding="utf-8"
    )

    fields = parser.extract_fields_from_record(record)
    fields = parser.prepare_fields(fields)
    rendered = PromptRenderer.render(template, fields)

    logger.debug("Rendered prompt:\n%s", rendered)

    (output_dir / "template_rendered_example.txt").write_text(
        rendered, encoding="utf-8"
    )


def save_rendered_prompt_debug(
    *,
    template: PromptTemplate,
    fields: dict,
    output_dir: Path,
    label: str = "",
):
    """Save a template and one rendered example for debugging.

    Lower-level variant for stages that manage their own prompts
    (no llm_task_cfg / parser needed).  *label* disambiguates when
    a stage uses multiple templates (e.g. "generate_true", "edit_to_false").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{label}_" if label else ""

    (output_dir / f"{prefix}template_used.txt").write_text(
        template.template_str, encoding="utf-8"
    )

    rendered = PromptRenderer.render(template, fields)

    (output_dir / f"{prefix}template_rendered_example.txt").write_text(
        rendered, encoding="utf-8"
    )
