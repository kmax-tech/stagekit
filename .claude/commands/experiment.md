# Fallacy-Detection Experiment Framework

You are operating inside a **reproducible ML experiment framework** for fallacy detection. The user wants to set up, modify, or debug an experiment. Use this reference to understand the architecture and produce correct configs and stages.

---

## Architecture Overview

The framework is a **DAG-based pipeline system** with declarative YAML configuration (Hydra), typed dataclass schemas, contract validation at build time, and full reproducibility tracking. Every experiment is a Cartesian product of dimensions (models, datasets, pipelines, modes) executed through a sequence of validated stages.

**Core principle**: Configuration defines the experiment. Stages are reusable building blocks. Contracts catch errors before execution. Artifacts are content-addressed and namespaced.

**Storage roles (one source of truth per question):**
- **Comparing runs/configurations** happens exclusively via `output/` + `RunIndex` + `pipeline_summary.json`, keyed by `repro_hash`. The hash covers the config AND the instrument (prompt versions + parser ids of the pipeline's stages, snapshotted into `config_used.yaml`; NOTES.md 2026-09-24). `output/` is the run ledger and is disposable.
- **`published/pins.yaml`** is the PUBLICATION layer: one pinned run per coordinate, the address a consumer YAML interpolates (`preload_artifacts: {key: pin:<coordinate>}`). The pin carries `run_dir`, `run_id`, `repro_hash` and a SHA-256 per artifact; the artifact is read from the run itself, nothing is copied. Never a comparison source. Pinning is an explicit act after a run (`python -m scripts.pin <run_dir> <coordinate>`; `--list` shows every pin's state); a missing pin is an error with candidates, never "the newest run". Pinned runs (`RunRecord.pinned`) are exempt from cleaning `output/`. Design: `docs/published-pins-statt-kopien.md`.

---

## Directory Layout

```
config/
  model/              # Model specs (e.g., gpt5-nano-model.yaml, llama3.2-model.yaml)
  llm_task/           # LLM task = task_name + template + parser (registry names)
  data/               # Dataset loaders with split config
  pipeline/           # Pipeline DAGs — stages + dependencies + feature config
    build/            #   dataset build pipelines (raw -> canonical -> classification)
    eval/             #   evaluation pipelines (bayes, zeroshot, bert, etc.)
    counterfactual/   #   counterfactual generation + classification
    casa/             #   CASA sufficiency (probability of sufficiency, Liu et al. 2024)
    tournament/       #   scenario tournament (l/w decomposition + G, Bradley-Terry log LR)
  experiments/        # Experiment definitions: Cartesian product of dimensions
                      #   (each sets its own max_workers)
  tracking/           # MLflow / W&B config

src/
  config/             # AppConfig dataclass hierarchy + path resolution
  pipeline/           # DAG executor, stage registry, data store
  stages/             # All execution stages (registered via @register_stage)
    canonical/        #   raw data -> CanonicalRecord converters
    derive/           #   data transformation stages (clean, validate, export)
    preprocessing/    #   preprocessing (hungarian matching)
  models/             # LLM backends (OpenAI, Ollama, claude/codex CLI) behind LLMBaseModel
  run/                # RunManager, pins (published/pins.yaml), reproducibility
  analysis/           # RunIndex (run discovery, repro_hash dedup), TableAggregator
  data_loader/        # Dataset loaders
  parsers/            # LLM response parsers (PARSER_REGISTRY)
  prompts/            # Prompt templates ($placeholder syntax, PROMPT_REGISTRY)

scripts/
  run_experiment.py   # THE entry point: loads experiment config(s), runs combos
  build_dataset.py    # Build canonical datasets from raw data
  debug.py            # Step-through debugging of any experiment combo
  analysis/, utils/   # Ad-hoc analysis scripts and combo filters

extensions/           # Parked ideas (decommissioned, no registry entries):
                      #   prompt_refinement, train_bert, unused canonical builders
```

There are NO per-experiment launcher scripts — `run_experiment.py` is the only runner.

---

## Config Hierarchy (stagekit/config/classes.py)

```
AppConfig (root)
  meta: MetaConfig         # run_id, experiment_name, run_group, cache, workers
  data: DataConfig         # dataset_path, data_name, split_id, mode (dataset|file|none)
  model: ModelConfig       # flat config for all model types: backend (openai|ollama|claude_cli|codex_cli),
                           # model_name, temperature (no subclasses)
  llm_task: LLMConfig      # task_name, template, parser
  pipeline: PipelineConfig # stages[], feature_sources, feature_views, preload_artifacts,
                           # preload_stages, source_modes, mode, variant,
                           # source_model, artifact_root
  training: TrainingConfig # batch_size, epochs, lr
  tracking: TrackingConfig # MLflow / W&B
```

### PipelineStageConfig
```python
@dataclass
class PipelineStageConfig:
    name: str                          # unique stage instance name
    stage_type: str = ""               # registered stage type (defaults to name if empty)
    model: str | None = None           # model YAML name (resolved at runtime)
    llm_task: str | None = None        # llm_task YAML name (resolved at runtime)
    config_override: dict | None = None  # stage-specific params (only DEVIATIONS from
                                         # the stage's code defaults)
    depends_on: list[str] | None = None  # DAG dependencies (stage names)
    model_cfg: ModelConfig | None = None       # resolved at runtime
    llm_task_cfg: LLMConfig | None = None      # resolved at runtime
```

---

## Stage System

### Registration
Stages are registered via two mechanisms:
1. the topic's `REGISTRATIONS["stage"]` in `src/fallacy_topic.py`:
   lazy-import lookup (stage_type -> module path)
2. `@register_stage("name")`: decorator on PipelineStage subclass

**To add a new stage**: create the class in the matching `src/stages/`
subpackage — `canonical/` (dataset canonicalizers), `derive/` (dataset
build operators), `classify/` (classifiers, kfold, eval_predictions),
`llm/` (LLM-driven stages), `preprocessing/`; shared infrastructure
(`base.py`, `artifact_*.py`) stays at the top level, beside
`variant_mapping.py` (the counterfactual story variants — topic code);
classifier infrastructure (`base`, `kfold_utils`,
`feature_utils`) lives inside `classify/`. Add `@register_stage("my_stage")`, add it to
`REGISTRATIONS["stage"]` in `src/fallacy_topic.py`, then follow the design
checklist below.

### Core and topic
The framework core (the list `CORE` in `tests/test_core_boundary.py`,
the part that leaves this repo as stagekit) never imports a fallacy
topic module; the test fails on any such import. Where the core needs
topic code, it asks a `LazyRegistry` and the topic registers there:
stages, prompts, parsers, data loaders, splits and models as above, plus
`DERIVATION_REGISTRY` (`stagekit/stages/derived_artifacts.py`, derivations of
a preloaded artifact), `PIPELINE_CHECK_REGISTRY`
(`stagekit/config/validation.py`, pipeline-level config checks) and
`RUN_LABEL_REGISTRY` (`stagekit/run/combinator.py`, short labels of an axis's
values in run names).

The topic enters by name: `config/config.yaml` lists it under `topics:`
(`src.fallacy_topic`), and `stagekit/run/topics.py` loads it before a config
is resolved, adding its `REGISTRATIONS` (registry kind -> {name: module})
to the registries. The framework's own maps (`STAGE_MODULE_MAP` and the
like) hold core modules only — splits, models, backends; a topic that
cannot be imported, registers under an unknown kind or maps a name
another topic or the framework maps elsewhere stops the load with its
name in the message. Not caught: mapping a name the framework registers
directly (e.g. the artifact schema `predictions`) — the framework's item
stays and the topic's entry is dead for that name. Kinds are written as
the registry's `kind`: lower-case words separated by spaces
(`REGISTRY_HOMES` in `stagekit/run/topics.py`).

From outside the repo, an experiment brings its own config directory and
prompts: `compose_app_config(overrides, config_dir=...)` makes that
directory Hydra's primary one (it needs its own `config.yaml`, which
names the topics) and puts `config/` behind it as `hydra.searchpath` — a
config there shadows the framework's of the same name, and the
composition, hence the repro hash, is otherwise unchanged.
`register_prompt_dir(path)` (`stagekit/prompts/registry.py`) registers every
`<name>.txt` there as prompt `<name>`, its `$placeholders` as required
fields; a prompt of a name the registry already has is refused, not
shadowed. Both are calls for a caller outside the repo (step 4a); nothing
in the repo passes `config_dir` yet, and `run_experiment` as well as the
repro hash of a config stored before topics existed read the topics of
`config/config.yaml` (`config_topics()`).

What a topic registers must fit: `stagekit/pipeline/conformance.py` has
checks for stages, prompts, parsers, prompt-parser pairs, artifact
schemas and data loaders — `check_stage`, `check_prompt`, `check_parser`
(+ `check_parser_cache_ids`), `check_prompt_parser_pair` (an llm_task),
`check_artifact_schema`, `check_data_loader` (on a sample) — each
returning its findings; derivations, pipeline checks and run-name labels
have none. `tests/test_conformance.py` runs them over every
registration of the fallacy topic; a new stage, prompt or parser is
covered there without a test of its own.

### Stage design checklist

Design rule of thumb: **everything the executor or a reader needs to
know about the stage is a declaration; everything that can be wrong is
checked at the earliest layer that can see it** (config < DAG build <
runtime), and a failure is always loud.

1. **Pick the base class** from the table below. Subclass an existing
   stage family before writing a bare `PipelineStage` — the k-fold and
   LLM bases own the fold loops, caching, batching, and prompt-debug
   mechanics.

2. **Declare the full I/O surface as class variables** —
   `requires_model/requires_data`, `required_artifacts` /
   `required_enrichments`, `provides_artifacts` /
   `provides_enrichments`, `_upstream_field`, and `publishes`
   ({field: filename}) if the stage's artifacts may be pinned as another
   pipeline's input (`stagekit/run/pins.py`, contributors listed in its
   `PUBLISHER_STAGE_TYPES`). The DAG executor and the
   preflight can only validate what is declared. Use the dynamic
   `get_provides_*` overrides ONLY when the key genuinely depends on
   resolved config (e.g. llm_task name) — never for convenience.

3. **Declare every config key: `required_config` + `optional_config`.**
   Every key the stage reads via `co.get(...)` goes in one of the two
   lists; undeclared keys in config_override are rejected at config
   time (typo protection, enforced by the base `check_stage_config`).
   Defaults live in stage code; YAML lists only deviations. Lists union
   over the MRO — declare only your own additions. Adding a new
   `co.get(...)` without extending the list fails preflight, so the
   declaration is part of the change.

4. **Stage-specific config rules go in `check_stage_config`** — always
   `errors = super().check_stage_config(stage_cfg, cfg)`, then extend.
   Do NOT edit `config_validation.py`; validation is dispatched to the
   stage class. Validate value constraints here too (enums, ranges) so
   they fail in preflight, not mid-run.

5. **If correctness depends on an upstream stage's CONFIG, check it at
   config time.** `check_stage_config` receives the full AppConfig —
   resolve the upstream via `depends_on` and validate compatibility
   (precedents: `hungarian_compute` requires its generate stage;
   `_check_variant_subset` in stages/llm/counterfactual.py requires the consumer's
   variants ⊆ producer's). Skip the check when the upstream is
   preloaded — that case is pinned (published/pins.yaml, `preload_expect`).

6. **Make artifacts self-describing.** A consumer must never re-derive
   semantics from names or conventions — put the fact in the artifact
   (precedent: story pairs carry `story_true`/`story_false` instead of
   consumers flipping `original`/`edited` by variant name). Declare an
   `artifact_schemas` entry for every `provides_artifacts` field; use a
   `custom_validator` for constraints the shape schema can't express.
   The same schema guards preloads, so legacy artifacts fail at the
   load boundary.

7. **Skips are data, total failure is an error.** Per-record skips go
   into a published artifact with a reason string and into
   StageResult.metrics (`skipped_by_reason`); a 100% failure of a
   requested unit (variant, fold, split) raises instead of warning
   (precedent: extract's coverage backstop). Never drop records
   silently — joins go through `join_by_id` / `_build_feature_matrix`
   with explicit `on_missing`, coverage into metrics.

8. **Reuse the shared utilities** — `serialization` (JSON/JSONL/YAML,
   `compute_file_hash`), `artifact_schemas.verify_recorded_hashes`,
   `classify.kfold_utils`, `classify.feature_utils`, `argument_text`. Rolling your own
   hashing/IO/join is a review defect. Prompts/parsers only via the
   registries.

9. **Respect run identity.** Any config field that cannot change the
   stage's output must be stripped in `repro_hash` (volatile lists) or
   live outside the identity (meta). Backend-dependent behavior is
   declared in `BACKEND_CAPABILITIES` next to the backend registry,
   never hardcoded elsewhere.

10. **Write the contract tests with the stage**: a declarations test
    (class vars as documented) and one test per cross-seam assumption —
    what the schema can't express, a test pins (precedent: the
    orientation test in test_llm_workflow_stage). Give the class its
    own docstring listing the config keys with defaults — the preflight
    DAG shows its first line as the node summary.

**Consuming artifacts in code/notebooks**: use the typed readers in
`stagekit/stages/artifact_readers.py` (`Scores`, `ScoresCV`, `Judgement`,
`Consensus` — `X.from_artifact(raw, source=...)`) instead of raw dict
lookups: they re-validate against the producer schema at read time and
give pyright-checked attribute access. Producers keep emitting plain
dicts — the schema stays the single contract.

### Stage Contract (PipelineStage base class)
Every stage declares:
```python
class MyStage(PipelineStage):
    requires_model: bool = False        # needs a loaded model
    requires_data: bool = False         # needs non-empty input data
    manages_own_prompts: bool = False   # True = stage sets parser/template itself
    required_artifacts: list[str] = []  # artifact keys from upstream
    required_enrichments: list[str] = [] # enrichment keys in records
    provides_enrichments: list[str] = [] # enrichment keys this stage writes
    provides_artifacts: list[str] = []   # artifact keys this stage produces
    required_config: list[str] = []      # config_override keys that MUST be set
                                         # (checked at config time AND runtime)
    optional_config: list[str] = []      # config_override keys read with a default;
                                         # required+optional (union over the MRO, see
                                         # allowed_config_keys()) are the ONLY keys
                                         # accepted in config_override — a typo fails
                                         # at config time instead of silently using
                                         # the default. "table_method" is allowed on
                                         # every stage (table row + baseline pin method).
    _upstream_field: str | None = None   # artifact field dependency

    @classmethod
    def check_stage_config(cls, stage_cfg, cfg) -> list[str]:
        # optional: stage-specific config rules (extend super()'s list)
        ...
```

### Base Classes — pick the right one

| Base | For | Subclass implements |
|---|---|---|
| `LLMGenerateStage` | one batch LLM call -> artifact (`{stage}.{output_field}`) | `output_field`, `desc`, `format_result`; opt. `select_records`, `task_fields`, `validate_outputs` |
| `LLMEnrichStage` | batch LLM call -> enrichments in records (provenance, `needs_enrichment` gating) | usually nothing (config-driven via llm_task) |
| `LLMWorkflowStage` | SEQUENCE of LLM steps with data flowing between them | `run()` calling `self.run_llm_step(model, prompt=..., parser=..., tasks=..., ctx=ctx)` per step; chaining is plain Python |
| `KFoldClassifierStage` | k-fold CV classifier (sklearn-style) | `prepare_inputs`, `run_fold`, `describe_model`, opt. `aggregate_extras`/`metrics_extras`/`metric_keys` |
| `TorchBertKFoldStage` | k-fold BERT/torch classifier (shared tokenize/train/predict fold) | `prepare_inputs` (state: texts/tokenizer/device), `build_classifier`, opt. `fold_features` |
| `PipelineStage` | everything else (compute, derive, canonical) | `run()` |

All LLM bases share one core (`LLMBatchStage`): select -> build tasks -> prompt debug -> batched call -> length check -> emit. Prompts/parsers are ALWAYS resolved from `PROMPT_REGISTRY`/`PARSER_REGISTRY` — never load prompt files or instantiate parsers inline in a stage.

### Stage Execution
```python
def run(self, model, data, ctx: StageContext) -> StageResult:
    # ctx.cfg: AppConfig
    # ctx.artifacts: dict of all upstream artifacts
    # ctx.stage_cfg: PipelineStageConfig
    # ctx.stage_dir: Path for this stage's outputs
    return StageResult(metrics={...}, artifacts={...}, data=None_or_replacement)
```

### Artifact Namespacing
All artifact keys MUST be `"{stage_name}.{field}"`. The executor merges into a global dict.

### Three Data Channels
1. **`record["enrichments"][key]`** — per-record, MODEL-produced intermediate results (LLM outputs, flags). Ground truth lives in `labels` only.
2. **`record["text"]`, `record["labels"]`** — stable canonical fields
3. **`ctx.artifacts["stage.field"]`** — aggregate stage outputs (predictions, feature dicts, scores)

### Available Stage Types

**Canonical loaders** (raw -> CanonicalRecord):
`fallacies_canonical`, `arg_microtexts_canonical`, `big_bench_canonical`,
`hong_canonical`, `sahai_canonical`, `touchefallacy_canonical`
(argotario/arg_tweets/mafalda/mm_used/nl2fol builders are parked in
`extensions/canonical_builders/` — not registered)

**Derive operators** (data transforms):
`drop_empty`, `therefore`, `build_components`, `mark_needs_enrichment`, `mark_cq_pilot`,
`balance_components`, `assemble_arguments`, `apply_premise_merge`, `find_collisions`,
`infer_claim_components`, `export_jsonl`, `validate_components`,
`validate_enrichments_integrity`, `validate_classification`,
`clean_meta`, `clean_circularity`, `clean_arguments`

**LLM stages** (generate artifacts or enrich records):
`llm_enrich` (per-record enrichment), `fallacies_binary` (zero-shot judge), `bayes_generate`,
`explanation_generate`, `explanation_combined_binary`, `explanation_binary_classify`,
`counterfactual_generate` (workflow: 4 chained steps), `counterfactual_evaluate` (workflow),
`hungarian_generate`,
`casa_claim_extract` (segment + pick the conclusion), `casa_intervene`
(negate, sample contexts, revise — CASA Tables 6-8; needs `negate`)
`tournament_scenario_sample` (k scenarios per side, fixed carriers merged),
`tournament_pair_judge` (sampled A/B votes over the l, w and g tours; anchors at
`anchor_votes_per_order`)

**Compute / evaluation**:
`casa_estimate` (NLI over CASA units, no LLM), `casa_score` (CASA verdicts,
diagnostics, stories.md),
`bt_score` (Bradley-Terry per tour → `log_lr_lw` / `log_lr_g` / `log_lr_l_only`,
`scenario_scores` table, role-keyed gate reported; kills only with `enforce_gates`),
`bayes_compute`, `hungarian_compute`, `counterfactual_extract` (NLI features),
`eval_predictions` (the ONLY scoring stage — all prediction artifacts feed it)

**Classification (k-fold CV)**:
`svm_clf` (SVM on numeric features), `bert_text_clf` (BERT text),
`bert_text_cq_clf` (BERT text + critical question), `bert_mlp_clf` (BERT+features),
`bert_gated_clf` (BERT+features with learned feature gate),
`stacked_logreg` (stacking meta-learner over sub-model prediction features)

**Parked (NOT registered — do not reference in pipelines)**: the
train_bert/apply_bert pair and the prompt_judge/judge_consensus stages
live in `extensions/` (hard-decommissioned 2026-07-23; see
extensions/README.md for revival).

**Utility**:
`kfold_split` (k-fold assignment generation)

---

## Pipeline YAML Structure

```yaml
name: my-pipeline-${pipeline.source_model}-${pipeline.mode}
mode: neutral                          # default parametric value
variant: both

# For pipelines whose LLM produces the enrichments: derive source_model
# from the runtime model — no per-model pipeline clones.
source_model: ${model.model_name}
# For pipelines consuming PRE-EXISTING enrichments: a literal default,
# overridden by the experiment's source_models axis.

# Two-tier feature selection (for classifier stages)
feature_sources:
  nli:
    - counterfactual_extract.nli_features
feature_views:
  nli_only: [nli]

# Pre-load artifacts from the pinned run (published/pins.yaml); the key is
# the artifact in that run, the value its coordinate
artifact_root: pin:counterfactual/${data.data_name}/${pipeline.source_model}/${pipeline.mode}/p${meta.prompt_nbr}
preload_artifacts:
  counterfactual_extract.nli_features: ${pipeline.artifact_root}

# Preloaded stages for pre-loaded artifacts (DAG nodes that don't execute)
preload_stages:
  counterfactual_generate:
    - counterfactual_generate.story_pairs

# Multi-mode feature stacking
source_modes: [neutral, skeptical, very_skeptical]

# Stage DAG — config_override lists ONLY deviations; hyperparameter
# defaults (epochs, batch_size, lr, ...) live in the stage classes.
stages:
  - name: kfold_split
    config_override: {folds: 5, seed: 42}
  - name: bert_text
    stage_type: bert_text_clf
    depends_on: [kfold_split]
    config_override:
      split_artifact: kfold_split.fold_assignments
      text_source: claim_premise
```

---

## Experiment YAML Structure

```yaml
# @package _global_
experiment_name: my-experiment
description: "What this experiment tests"

max_workers: 1        # 1 = local CV / ollama; 50-100 = API concurrency

# Cartesian product dimensions (all optional, default to [None]).
# To compare models, list several — no extra files needed:
models:
  - gpt5-nano-model
  - llama3.2-model

data: [hong-clf]
pipelines: [counterfactual/generate]
llm_tasks: [bayes-neutral-evidence-llm]

# Pipeline-parametric axes
modes: [neutral, skeptical]
variants: [true_to_false, both]
source_models: [gpt-5-nano-2025-08-07]   # only for pipelines with a literal source_model

# Optional
prompt_nbrs: [1, 2]
split_id: "no_split"
```

The runner generates all combos via Cartesian product, composes Hydra config per combo, checks RunIndex (`repro_hash`) for already-completed equivalent runs, executes the pipeline, and prints the `scripts.pin` command when the run offers pinnable artifacts — pinning itself is the operator's decision.

---

## Running Experiments

```bash
# List all experiments
python -m scripts.run_experiment --list

# Run one experiment (max_workers from its YAML)
python -m scripts.run_experiment counterfactual/hong-generate-exp

# Run several sequentially
python -m scripts.run_experiment eval/hong-eval-cv-exp counterfactual/hong-svm-exp

# Overrides
python -m scripts.run_experiment eval/bert-text-exp --max-workers 50 --force

# Preflight: validate every combo + print the pipeline DAG (ASCII + Mermaid),
# nothing executes, nothing is written. Exit code 1 if any combo fails.
python -m scripts.run_experiment eval/bert-text-exp --dry-run

# Build a canonical dataset
python -m scripts.build_dataset hong
```

---

## Prompts & LLM tasks

- Every prompt is a **registered template**: `.txt` file under `src/prompts/` + entry in
  `REGISTRATIONS["prompt"]` (`src/fallacy_topic.py`) + `@register_prompt` factory in `src/prompts/catalog.py`
  (declares `required_fields` once). `$placeholder` syntax.
- Single-step stages pair prompt+parser via a `config/llm_task/*.yaml`
  (`task_name`, `template`, `parser`).
- Multi-step stages (LLMWorkflowStage) resolve each step by registry name:
  `self.run_llm_step(model, prompt="counterfactual_edit_to_false", parser="counterfactual_edit", tasks=..., ctx=ctx)`.
- NEVER build prompts as inline Python strings or read prompt files directly in a stage.
- LLM cache key = (prompt text, model, temperature, parser class, template hash) — cached
  under `data/cache/llm/<experiment_name or task_name>/`. Changing prompt text invalidates.
- `meta.cache_mode`: `normal` (a miss calls the backend, the answer is stored) or
  `replay` (every answer from the cache; no backend built or called; a miss raises
  `CacheMiss` naming item and key, except items listed in `meta.replay_failed_items` —
  the replayed run's own failed items — which count as failed). The mode, the list and
  `meta.labkit_run_id` land in `pipeline_summary.json`; none enters the repro hash.
- Backend `labkit` (`stagekit/models/backends/labkit_backend.py`): `model_name` is a
  labkit `models.toml` alias; calls go through labkit's cache of raw answers when the
  caller binds one (`answers_through(Cache(run, replay=…))`), otherwise uncached. Run it
  with `meta.enable_cache=false` (no second, parsed store) and `meta.max_retries=0` (a
  retry would read the same stored answer); the cache mode is in the labkit record.

---

## How to Set Up a New Experiment

### Step 1: Ensure data exists
Either use existing data in `config/data/` or create a new canonical loader stage + build pipeline.

### Step 2: Create/select a pipeline
Define a pipeline YAML in `config/pipeline/{category}/`. Each stage must:
- Reference a registered `stage_type` (or have `name` matching a registered type)
- Declare `depends_on` for correct execution order
- Provide required `config_override` keys (see the stage's `required_config`)

### Step 3: Create experiment config
Define an experiment YAML in `config/experiments/{category}/` with the Cartesian
product dimensions and `max_workers`. Prefer adding a model/mode to an existing
experiment's axes over cloning files.

### Step 4: Validate
- Every `stage_type` must be known to `STAGE_REGISTRY` (a loaded topic maps it)
- Every `model`/`llm_task`/`data` reference must have a matching YAML
- DAG dependencies must be acyclic
- Artifact keys must use `"{stage_name}.{field}"` namespacing
- Stage config rules are enforced by each class's `check_stage_config`
- Run it all at once with `python -m scripts.run_experiment <name> --dry-run`
  (`stagekit/pipeline/preflight.py`): composes every combo, runs the executor's
  build-time checks (contracts, cycles, preload hashes), prints the DAG —
  without executing a stage or writing to `output/`

### Step 5: Run
```bash
python -m scripts.run_experiment {category}/{experiment-name}
```

---

## Key Conventions

1. **YAML filenames use hyphens** (`bayes-neutral-evidence-llm.yaml`), **Python identifiers use underscores** (`stage_type: bayes_generate`)
2. **Artifacts are namespaced**: `"{stage_name}.{field}"` — never bare keys
3. **Enrichments vs artifacts**: model output consumed by a LATER stage as part of the record goes in `record["enrichments"]`; the experiment's scored/compared result (feature vectors, predictions) goes in artifacts. Gold truth is `labels`, written only by build pipelines.
4. **Binary labels**: always use `get_binary_label(labels)` from `canonical/schema.py`, never hardcode label keys
5. **Parametric interpolation**: use `${pipeline.mode}`, `${pipeline.variant}`, `${model.model_name}`, `${data.data_name}`, etc. in YAML
6. **Model choice is a parameter, not a file**: never clone a pipeline/experiment YAML to swap a model — use the `models:` axis (+ `source_model: ${model.model_name}` in LLM pipelines)
7. **Hyperparameter defaults live in stage code**; YAML `config_override` lists only deviations
8. **Prompts live in the registry** — no inline prompt strings, no direct file loads in stages
9. **Compare runs only via RunIndex/`repro_hash`**; `published/pins.yaml` names the input runs, it is not a comparison source
10. **Legacy data is expendable**: don't compromise new design for backward compatibility
11. **Tests**: always run `python -m pytest tests/ -v` after code changes

---

## When the User Asks You To...

**Add a new dataset**: Create a canonical stage in `src/stages/canonical/`, register it, create a build pipeline YAML, create a data config YAML, then create a build experiment config.

**Add a new evaluation method**: Create a new stage in `src/stages/` (pick the right base class from the table above), register it, wire it into a pipeline YAML (with correct depends_on), create or extend an experiment config.

**Add a new classifier**: Subclass `KFoldClassifierStage` (or `TorchBertKFoldStage` for BERT-based), implement the hooks, declare `required_config`/`check_stage_config`, use `resolve_feature_keys()` for feature resolution, wire into a pipeline with `kfold_split` upstream. Feature joins are explicit: records with missing/incomplete features are DROPPED by default (`on_missing_features: drop | fail | zero` in config_override) — never silently zero-filled.

**Add a multi-step LLM workflow**: Subclass `LLMWorkflowStage`; register each step's prompt in `src/prompts/catalog.py` and parser in `stagekit/parsers/registry.py`; call `run_llm_step` per step and chain results in plain Python.

**Refine a prompt (error-driven, arginvariant-style)**: parked in `extensions/prompt_refinement/` (decommissioned 2026-07-23, see `extensions/README.md` for revival). Already-generated prompt versions in `src/prompts/generated/<base>/v<N>.txt` stay registered as `<base>__gen_v<N>` and remain drop-in usable via the `llm_task.template` override.

**Try a new model on an existing setup**: Add it to the experiment's `models:` list (create a `config/model/*.yaml` if new). Do not clone pipelines or experiments.

**Change feature views**: Edit the `feature_sources` and `feature_views` in the pipeline YAML. No code changes needed.

**Pre-load artifacts from a previous run**: pin the run (`python -m scripts.pin <run_dir> <coordinate>`), then set `preload_artifacts: {artifact_key: pin:<coordinate>}` and optionally `preload_stages` in the pipeline YAML. Config validation checks that the pin exists, its run is on disk and carries the key; the executor verifies the artifact's hash against the pin and records the pins in the run's `preload_provenance.json`. Something the consumer computes FROM a pinned artifact (the baseline labels as a numeric feature, likelihoods from probabilities) is declared as `derived_artifacts: {derived.<field>: {from: <preloaded key>, derive: <name>}}` (`stagekit/stages/derived_artifacts.py`) — the derivation is this run's work, never the pin's; when stacking it runs per pinned mode.

**Debug a failed stage**: Check the stage's `required_artifacts`, `required_enrichments`, and `required_config` declarations against what the pipeline provides. The executor logs contract violations before execution; config-time errors come from the stage's own `check_stage_config`.
