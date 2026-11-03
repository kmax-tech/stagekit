import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------
# Dataset Configuration
# --------------------------------------------------------
@dataclass
class DataConfig:

    # No default: the topic's config.yaml sets it (validation refuses "").
    data_loader: str = ""
    dataset_path: str = "default_dataset"
    data_name: str = "default_name"
    label_info_path: Optional[str] = None
    split_id: str = "no_split"
    split_dir: str = "split"

    # Unified input config
    mode: str = "dataset"  # dataset | file | none
    file_path: Optional[str] = None


# --------------------------------------------------------
# Model Configuration
# --------------------------------------------------------
@dataclass
class ModelConfig:
    """Flat model config for all model types (LLM, transformer, etc.).

    One flat class because Hydra validates YAML keys against the type
    registered in AppConfig — a polymorphic ``model`` field would
    reject subclass-only keys at compose time. Keep this dataclass
    minimal: every field here enters config_used.yaml and (unless
    excluded in repro_hash._model_canon) the run identity.
    """
    model_type: str = "llm"
    backend: str = ""          # ollama | openai | vllm | claude_cli | codex_cli
    model_name: str = ""
    model_dir: Optional[str] = None
    description: str = ""
    temperature: float = 1     # per-backend effect: see BACKEND_CAPABILITIES
    # Whether the SERVED MODEL accepts a temperature override (some hosted
    # models, e.g. gpt-5-nano/-mini, reject or ignore it). Declared in the
    # model YAML — one truth per model, instead of a hardcoded ID list in
    # the backend that silently missed new snapshots. When False,
    # temperature drops out of run identity and cache key (it cannot
    # affect output), and validation rejects a config that sets one.
    supports_temperature: bool = True
    # Token budget per answer; None means "no budget", which is what every
    # stored run used. A reasoning model needs enough for trace AND answer —
    # the trace burns the budget first.
    max_tokens: Optional[int] = None
    # Whether this model emits a thinking trace. On an OpenAI-compatible
    # server this demands `vllm serve --reasoning-parser ...`; declaring it
    # is what lets the backend refuse an unconfigured server instead of
    # feeding the trace to a parser. See NOTES.md 2026-09-01.
    reasoning: bool = False
    # Raw-completion knobs, consumed by `ollama_completion` only: sequences
    # that end the answer, and the context window in tokens (ollama's
    # default depends on the server version and it truncates a long prompt
    # from the front, silently). None for every stored run; they enter run
    # identity only where a backend declares them (BACKEND_CAPABILITIES).
    stop: Optional[List[str]] = None
    num_ctx: Optional[int] = None



# --------------------------------------------------------
# LLM Configuration
# --------------------------------------------------------
@dataclass
class LLMConfig:
    task_name: str = ""
    template: Optional[str] = None
    parser: Optional[str] = None

# --------------------------------------------------------
# Tracking Configuration
# --------------------------------------------------------
@dataclass
class TrackingConfig:
    enable_mlflow: bool = False
    enable_wandb: bool = False

    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = ""

    wandb_project: str = ""
    wandb_entity: Optional[str] = None
    wandb_mode: str = "offline"  # online | offline | disabled


# --------------------------------------------------------
# Pipeline Stage Configuration
# --------------------------------------------------------
@dataclass
class PipelineStageConfig:
    name: str
    stage_type: str = ""
    # Raw config names from YAML
    model: str | None = None
    llm_task: str | None = None

    # Stage-local configuration override (e.g., hyperparameters)
    config_override: dict | None = None

    # DAG
    depends_on: list[str] | None = None

    # -------------------------------
    # RESOLVED OBJECTS (filled later)
    # -------------------------------
    model_cfg: "ModelConfig | None" = None
    llm_task_cfg: "LLMConfig | None" = None


# --------------------------------------------------------
# Pipeline Configuration
# --------------------------------------------------------
@dataclass
class PipelineConfig:
    name: str = ""
    stages: List[PipelineStageConfig] = field(default_factory=list)

    # Parametric axes — used via OmegaConf interpolations
    # (${pipeline.mode}, ${pipeline.variant},
    #  ${pipeline.source_model}, ${pipeline.method},
    #  ${pipeline.artifact_root}) in
    # preload_artifacts values (pin coordinates) and stage config_override values.
    mode: str = ""
    variant: str = ""  # no default: the topic's config.yaml sets it
    source_model: str = ""
    method: str = ""
    artifact_root: str = ""

    # Two-tier feature selection for classifier stages.
    # feature_sources: semantic name → physical artifact keys
    # feature_views:   experiment name → list of source names
    # feature_source_scopes: source name → "mode" | "global"
    #   "mode" (default): artifacts are mode-specific, loaded per mode
    #   "global": artifacts are mode-independent, loaded once
    # Stages reference a view via config_override.feature_view;
    # the resolver in fallacy-detection, src/stages/classify/feature_utils.py, expands it.
    feature_sources: Dict[str, List[str]] = field(default_factory=dict)
    feature_source_scopes: Dict[str, str] = field(default_factory=dict)
    feature_views: Dict[str, List[str]] = field(default_factory=dict)

    # Explicit preloaded DAG nodes — stages that don't execute but whose
    # artifacts are pre-loaded from disk.  Each key is a stage name that
    # may appear in depends_on; the value lists the artifact keys it
    # provides (same "stage.field" convention as real stages).
    #
    # Example:
    #   preload_stages:
    #     counterfactual_generate:
    #       - counterfactual_generate.story_pairs
    preload_stages: Dict[str, List[str]] = field(default_factory=dict)

    # Multi-mode feature stacking: load features from multiple
    # skepticism modes and concatenate into one feature vector.
    # When non-empty, classifier stages load artifacts from each
    # listed mode directory instead of using single-mode preloads.
    # Feature names are namespaced as {mode}_{feature_name}.
    source_modes: List[str] = field(default_factory=list)

    # Pre-load artifacts from pinned runs instead of computing them.
    # Maps artifact keys (e.g. "counterfactual_extract.nli_features")
    # to "pin:<coordinate>" — the pin in published/pins.yaml names the
    # run, the key is read from it (stagekit/run/pins.py). File paths are
    # not accepted. Every key declared in preload_stages.provides must
    # appear here.
    preload_artifacts: Dict[str, str] = field(default_factory=dict)

    # Artifacts THIS run derives from a preloaded one, with a named
    # transform (stagekit/stages/derived_artifacts.py):
    #   derived_artifacts:
    #     derived.cf_likelihoods:
    #       from: counterfactual_evaluate.probabilities
    #       derive: cf_likelihoods_from_probabilities
    # Materialised after preloading and per pinned mode when stacking;
    # the derivation is this run's work, so pins stay pure pointers.
    derived_artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Hard expectations on the pins loaded from.  If set, the executor
    # checks every pin against these and raises on mismatch: run_id,
    # repro_hash (scalars) and artifact_hashes ({artifact_key: sha256}).
    preload_expect: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------
# Meta Config
# --------------------------------------------------------
@dataclass
class MetaConfig:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = ""
    run_name: str = ""
    run_group: str = "default"
    experiment_name: str = ""

    # Hydra config group names — set by the experiment runner so
    # run_inputs.yaml can trace which YAMLs produced this run.
    experiment_config: str = ""
    data_config: str = ""
    pipeline_config: str = ""

    enable_cache: bool = True
    cache_dir: str = "data/cache/llm"
    # How the LLM cache is used: "normal" | "replay"
    # (stagekit/models/llm_cache.py, CACHE_MODES). A replay needs the cache.
    cache_mode: str = "normal"
    # Replay only: items the replayed run itself lists as failed (its
    # llm_stats failed_items). Their misses count as failed items instead
    # of stopping the run. Passed by the caller, never written in stagekit.
    replay_failed_items: List[str] = field(default_factory=list)
    # The labkit run this pipeline runs inside, if any; written to
    # pipeline_summary.json so the two records name each other.
    labkit_run_id: str = ""
    prompt_nbr: int = 1
    max_workers: int = 50
    max_retries: int = 10
    save_raw: bool = True
    store_parse_diagnostics: bool = True
    store_only_failed_parses: bool = True
    enable_pipeline_outputs: bool = True
    max_parse_diagnostics_per_stage: Optional[int] = None

# --------------------------------------------------------
# Fusion Configuration
# --------------------------------------------------------
# --------------------------------------------------------
# Top-level Config
# --------------------------------------------------------
@dataclass
class AppConfig:
    meta: MetaConfig = field(default_factory=MetaConfig)

    # Topic modules whose registrations this run needs (stagekit/run/topics.py);
    # code to load, not identity — the repro hash leaves them out.
    topics: List[str] = field(default_factory=list)

    output_dir: str = "output"

    random_state: int = 42
    model: Optional[ModelConfig] = field(default_factory=ModelConfig)
    llm_task: Optional[LLMConfig] = None
    data: DataConfig = field(default_factory=DataConfig)

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict:
        """Convert config (with nested dataclasses) into a pure dictionary."""
        return asdict(self)
