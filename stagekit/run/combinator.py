from itertools import product
from typing import Sequence
from omegaconf import DictConfig, ListConfig

from stagekit.utils.registry import LazyRegistry

#: axis → module that registers it; the registered item is a mapping
#: {value: short label} for run names (not a function), owned by the
#: topic that defines the axis's values. The framework has none of its
#: own; a topic adds its entries when loaded (stagekit/run/topics.py).
RUN_LABEL_MODULE_MAP: dict[str, str] = {}

RUN_LABEL_REGISTRY = LazyRegistry("run-name label", RUN_LABEL_MODULE_MAP)


def get_dimension(
    cfg: DictConfig,
    key: str,
    *,
    default: Sequence[object] | None = None,
) -> Sequence[object]:

    if key not in cfg:
        return default if default is not None else ()

    value = cfg[key]

    if isinstance(value, DictConfig):
        raise TypeError(
            f"Experiment dimension '{key}' must be a scalar or list, "
            f"not a mapping"
        )

    # handle OmegaConf lists + Python sequences
    if isinstance(value, (list, tuple, ListConfig)):
        return tuple(value)

    return (value,)

def generate_combos(exp_cfg: DictConfig):
    """
    Generate experiment combinations from a DictConfig.

    Missing dimensions default to a single None value, meaning
    'no variation on this axis'.
    """

    # Core dimensions (safe even if missing)
    models = get_dimension(exp_cfg, "models", default=(None,))
    data_items = get_dimension(exp_cfg, "data", default=(None,))
    pipelines = get_dimension(exp_cfg, "pipelines", default=(None,))
    llm_tasks = get_dimension(exp_cfg, "llm_tasks", default=(None,))
    prompt_nbrs = get_dimension(exp_cfg, "prompt_nbrs", default=(None,))

    # Pipeline-parametric axes — forwarded as pipeline.X overrides
    modes = get_dimension(exp_cfg, "modes", default=(None,))
    variants = get_dimension(exp_cfg, "variants", default=(None,))
    source_models = get_dimension(exp_cfg, "source_models", default=(None,))
    methods = get_dimension(exp_cfg, "methods", default=(None,))

    for model, llm_task, data, pipeline, prompt_nbr, mode, variant, source_model, method in product(
        models,
        llm_tasks,
        data_items,
        pipelines,
        prompt_nbrs,
        modes,
        variants,
        source_models,
        methods,
    ):
        combo_dict: dict[str, object] = {
            "experiment_name": exp_cfg.experiment_name,
            "model": model,
            "llm_task": llm_task,
            "data": data,
            "pipeline": pipeline,
            "mode": mode,
            "variant": variant,
            "source_model": source_model,
            "method": method,
        }

        # Pass through optional scalar config
        if "split_id" in exp_cfg:
            combo_dict["split_id"] = exp_cfg.split_id

        name_parts: list[str] = []

        if model is not None:
            name_parts.append(str(model))
        if llm_task is not None:
            name_parts.append(str(llm_task))
        if data is not None:
            name_parts.append(str(data))
        if pipeline is not None:
            name_parts.append(str(pipeline))
        if prompt_nbr is not None:
            combo_dict["prompt_nbr"] = prompt_nbr
            name_parts.append(f"p{prompt_nbr}")
        if mode is not None:
            name_parts.append(str(mode))
        if variant is not None:
            labels: dict[str, str] = RUN_LABEL_REGISTRY.get("variant")
            short = labels.get(str(variant), str(variant))
            name_parts.append(f"v{short}")
        if method is not None:
            name_parts.append(str(method))

        combo_dict["name"] = "-".join(name_parts) if name_parts else "default"

        yield DictConfig(combo_dict)
