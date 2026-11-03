import functools
from datetime import datetime
import uuid
from stagekit.config.classes import AppConfig
import stagekit.config.paths as s
from pathlib import Path
from stagekit.run.context import RunContext
from stagekit.run.topics import load_topics
from stagekit.config.validation import validate_runtime_config


def _compose_root(overrides: list[str], config_dir: Path | None):
    """Hydra compose of ``config`` — the one place the search path is set.

    Without *config_dir* the framework's config directory is the only
    one. With it, *config_dir* is the primary directory and must hold its
    own ``config.yaml`` (it names the topics); the framework's directory
    follows as ``hydra.searchpath``. Hydra searches the primary directory
    first, so a config there shadows the framework's of the same name.
    """
    from hydra import compose, initialize_config_dir
    import stagekit.config.schema  # noqa: F401 — registers the schema config.yaml names

    if config_dir is None:
        primary, extra = s.CONFIG_DIR, []
    else:
        if not (config_dir / "config.yaml").is_file():
            raise FileNotFoundError(
                f"{config_dir} has no config.yaml — an experiment's config "
                f"directory names its topics there"
            )
        primary = config_dir.resolve()
        extra = [f"hydra.searchpath=[file://{s.CONFIG_DIR}]"]
    with initialize_config_dir(config_dir=str(primary), version_base="1.3"):
        return compose(config_name="config", overrides=[*overrides, *extra])


def compose_app_config(overrides: list[str], *, config_dir: Path | None = None) -> AppConfig:
    """THE config-composition sequence — single source of truth.

    Hydra compose of the root ``config`` with *overrides*, converted to
    a validated AppConfig with resolved paths. Every runner (experiment,
    baseline, dataset build, story generation, debug, refinement) calls
    this instead of hand-rolling initialize/compose/convert/configure —
    a Hydra API or config-name change happens in exactly one place.
    *config_dir*: an experiment's own config directory, searched before
    the framework's (``_compose_root``).
    """
    from stagekit.config.resolution import dictcfg_to_appcfg

    cfg_dictcfg = _compose_root(overrides, config_dir)
    # Before conversion: validation looks names up in the registries.
    load_topics(list(cfg_dictcfg.topics))
    return configure_paths(dictcfg_to_appcfg(cfg_dictcfg))


@functools.cache
def config_topics(config_dir: Path | None = None) -> tuple[str, ...]:
    """The topics the root config names (``topics:`` in config.yaml).

    For code that needs the registries before or without composing a run
    config: the experiment runner before it builds combos, the repro hash
    of a stored config that names no topics.
    """
    return tuple(_compose_root([], config_dir).topics)


def compose_experiment_config(experiment_name: str):
    """Compose an experiment definition from config/experiments/."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(s.EXPERIMENTS_CONFIG_DIR), version_base="1.3",
    ):
        return compose(config_name=experiment_name)


def build_run_group(
    experiment_name: str,
    *,
    model: str | None = None,
    llm_task: str | None = None,
    data: str | None = None,
    pipeline: str | None = None,
    prompt_nbr: int | str | None = None,
    mode: str | None = None,
    variant: str | None = None,
    source_model: str | None = None,
    method: str | None = None,
) -> str:
    """Build a run_group string from all varying axes of an experiment.

    Only axes that are actually set (not None) are included.  The result
    is a ``__``-separated string like
    ``experiment_name__model__data__pipeline``.

    This replaces ad-hoc inline construction in the experiment runners so
    that logically different combos always land in distinct groups.
    """
    parts = [experiment_name]
    for value in [model, llm_task, data, pipeline, prompt_nbr, mode, variant, source_model, method]:
        if value is not None:
            parts.append(str(value))
    return "__".join(parts)


def configure_paths(cfg: AppConfig):
    """
    Ensures project_root, output_dir, and data split paths
    are fully resolved before anything uses them.

    Runs ``validate_runtime_config`` after resolution to catch
    missing files/directories early.
    """

    project_root = s.PROJECT_ROOT

    if not cfg.output_dir:
        cfg.output_dir = str(project_root / "output")

    validate_runtime_config(cfg)

    return cfg



def make_run_context(
    cfg: AppConfig,
    *,
    run_type: str,
    group: str | None = None,
) -> RunContext:
    base = Path(cfg.output_dir) / run_type
    if group:
        base = base / group

    run_dir = base / cfg.meta.run_name

    return RunContext(
        run_dir=run_dir,
        run_type=run_type,
        run_name=cfg.meta.experiment_name
    )


def add_metadata(cfg: AppConfig):
    """
    Adds timestamp + UUID to cfg.meta.
    Does NOT handle run_name — that is done separately.
    """

    cfg.meta.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.meta.run_id = str(uuid.uuid4())



def add_run_name(cfg: AppConfig) -> None:
    """Set the run directory name: experiment + timestamp + run_id prefix.

    The timestamp alone has one-second resolution, so two runs of the
    same combo within a second (tests, quick --force reruns) landed in
    the SAME directory and interleaved their stage outputs (audit C5).
    The run_id suffix makes the name collision-free without touching the
    group layout (combo dirs are the parent, not the leaf).

    Raises:
        ValueError: If ``add_metadata`` has not run yet — the name would
            silently lose its timestamp (run_id has a dataclass default,
            the timestamp does not).
    """
    if not cfg.meta.timestamp or not cfg.meta.run_id:
        raise ValueError(
            "add_run_name requires add_metadata first (timestamp/run_id unset)"
        )
    short_id = cfg.meta.run_id[:8]
    cfg.meta.run_name = f"{cfg.meta.experiment_name}_{cfg.meta.timestamp}_{short_id}"