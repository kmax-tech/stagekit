# Notes — hard problems and why they were solved this way

Short records of non-obvious decisions. The code shows *what* works;
this file preserves the *why* and the approaches that did not.

## 2026-07-03 — LLM cache stability when migrating prompts into the registry

**Problem:** Moving inline prompt loading (counterfactual stages, clean_*
SYSTEM_PROMPT strings) into PROMPT_REGISTRY risked invalidating all LLM
caches: the cache key includes the exact prompt text, the parser class
name, and the template hash (`PromptTemplate.version` = sha of
template_str).

**Solution:** Registry factories load the *same* .txt files (identical
template_str → identical version) and resolve the *same* parser classes.
For the clean_* stages the extracted .txt templates were verified
byte-identical to the old f-string assembly before switching.

**Notes:** Any prompt refactor must keep rendered text byte-identical or
accept a full cache miss. `prompt_version` changes whenever template_str
changes — even whitespace.

## 2026-07-03 — Cache scoping: experiment_name vs task_name

**Problem:** LLM cache dirs under `data/cache/llm/` use two schemes:
generate stages scope by `meta.experiment_name`, enrich stages by the
llm_task's `task_name`. Renaming an experiment therefore orphans the
generate caches.

**Solution:** When experiments were renamed (gptoss-suffix removal), the
cache directories were `git mv`'d to the new experiment names; enrich
caches (task-scoped) were unaffected.

**Notes:** Renaming an experiment? Move `data/cache/llm/<old-name>/` too.
The scoping lives in `LLMBatchStage.cache_dataset_id` /
`LLMEnrichStage.cache_dataset_id`.

## 2026-07-03 — repro_hash is the one comparison key; artifacts/ is publication

**Problem:** Three identity schemes coexisted (RunIndex repro_hash,
ArtifactIndex semantic tuple, exported content_hash+path) with no shared
key; ArtifactIndex duplicated RunIndex over the same output/ tree with
zero production consumers.

**Solution:** Comparing runs happens only via output/ + RunIndex +
pipeline_summary.json, keyed by repro_hash. artifacts/ is a read-only
publication layer for pipeline inputs; every export's metadata.json now
carries the source run's repro_hash (backfilled via
scripts/backfill_repro_hash.py). ArtifactIndex was deleted.

**Notes:** output/ is disposable, artifacts/ is pinned. Do not resurrect
an artifact index over output/ — extend RunIndex if run-level metadata
is needed.

## 2026-07-03 — Join semantics: drop, not zero-fill

**Problem:** `_build_feature_matrix` silently zero-filled records with
missing/incomplete features; classifiers trained on zero vectors with
only a log warning. The classic silent failure.

**Solution:** Explicit `on_missing: drop | fail | zero` (default drop)
plus `join_by_id`/`JoinReport` for all artifact↔record joins — coverage
is published into StageResult.metrics (e.g. `prediction_join_coverage`)
so gaps appear in run summaries.

**Notes:** Multi-mode fusion stacking keeps `zero` deliberately (partial
per-mode coverage is by design). Default change means runs with feature
gaps now train on fewer records than before — that is intended.

## 2026-07-03 — sklearn typing/behavior drift

**Problem:** (a) sklearn stubs type `zero_division` as `str` although 0
is valid at runtime — pyright false positives. (b) `roc_auc_score` on a
single-class fold raises ValueError in older sklearn but returns nan
with a warning in newer versions, so an `except ValueError` fallback
silently propagated nan into metric means.

**Solution:** (a) targeted `# pyright: ignore[reportArgumentType]` with a
"sklearn stub gap" note at the call sites. (b) `binary_fold_metrics`
catches ValueError *and* maps nan → 0.0.

## 2026-07-03 — Pyright scoping for duck-typed test fakes

**Problem:** Tests use plain fake classes (`_FakeModel`) where stages
expect `LLMBaseModel` — idiomatic duck typing, but pyright basic mode
reported ~150 errors in tests alone.

**Solution:** `pyrightconfig.json` defines an executionEnvironment for
`tests/` with the duck-typing rules disabled; `src/` stays strict.
Gotcha: the environment needs `"extraPaths": ["."]`, otherwise `src.*`
imports stop resolving inside it.

## 2026-07-03 — Cross-process cache safety via sidecar flock

**Problem:** Running two experiments in parallel processes on the SAME
model raced on the LLM cache's shard JSONs: both read-modify-wrote the
file, the last writer dropped the other's entries (measured: ~50% lost
updates under contention). Corruption was never possible (atomic
temp+rename writes) — only lost entries, i.e. paid re-computation.

**Solution:** `get_or_set` and `_store_prompt` now hold an
`fcntl.flock` in addition to the thread lock, and re-read the shard
under both locks so concurrent writers merge. The lock lives on a
SIDECAR file (`<shard>.lock`, gitignored) because `os.replace` swaps
the shard's inode — locking the data file itself would stop protecting
it after the first write.

**Notes:** Reads take no file lock on purpose (stale read = miss =
harmless). Cache data format/keys/layout are untouched; existing
git-committed caches stay byte-identical. A central cache server was
considered and rejected: it only adds value for cross-machine sharing
or in-flight request coalescing, neither of which applies here.

**Known limit — local disk only:** `fcntl.flock` is unreliable on
network filesystems (NFS et al.). This design assumes the cache lives
on a LOCAL disk. If the project ever moves to a cluster with shared
storage, do not trust the flock — migrate the cache backend to SQLite
(one DB per model, WAL mode), which handles cross-process/cross-host
locking properly. Same trigger applies at ~10-100x today's cache size:
each write rewrites the whole shard JSON (O(shard), not O(entry)), so
large shards make SQLite or append-only logs the better backend.

## 2026-07-04 — One F1 vocabulary, one fold generator, one compose sequence

**Problem:** A post-refactor audit found three silent-inconsistency
sources: (a) eval_predictions stored BINARY F1 in the scores artifact
while the refinement driver recomputed MACRO F1 from the same
per_record — two numbers both called "f1"; (b) the CV fold algorithm
existed twice (kfold_split._create and kfold_utils.resolve_folds'
fallback) — editing one side would make runs with and without a split
artifact evaluate on different folds; (c) the Hydra
compose→AppConfig→configure_paths sequence was copy-pasted across six
runners.

**Solution:** eval_predictions now stores BOTH "f1" (binary, matching
the classifier stages) and "f1_macro"; the refinement driver reads
f1_macro from the artifact and never recomputes (loop.macro_f1
deleted). kfold_utils.make_fold_assignments is THE fold generator —
kfold_split and the on-the-fly fallback both call it; shared
test_ids_by_fold/aggregate_fold_metrics replace the hand-rolled copies
in eval_predictions and classify_base. run_utils.compose_app_config /
compose_experiment_config replace all six inline compose blocks;
CONFIG_DIR/EXPERIMENTS_CONFIG_DIR live in stagekit/config/paths.py.

**Notes:** Dead parallel implementations deleted in the same sweep:
TransformerModel (stagekit/models/bert.py + bert-model.yaml — second BERT
training path, zero consumers), train_test/train_dev_test split
strategies + create_split.py, utils/evaluation.py. TrainingConfig +
config/training/ removed — it was doubly dead (not in config.yaml
defaults AND stages read config_override["epochs"], not num_epochs);
hyperparameter defaults live in stage code per convention. Removing
the training block from the repro canon changes repro_hash for ALL
runs again (accepted, legacy expendable). mlp_bert_cls now honors
hidden_dim like its gated sibling.

## 2026-07-04 — Vocabulary: enrichments, not annotations

**Problem:** The record namespace `annotations.*` collided with the NLP
convention (annotation ≈ curated gold labels) while actually holding
the opposite — model-produced, per-run intermediate results — and the
producing stage family was named "enrich", so one concept carried two
names (enrich writes annotations).

**Solution:** Unified on the enrich family: record namespace is now
`enrichments.*`, stage contracts are provides_enrichments /
required_enrichments, the gate stage is mark_needs_enrichment
(needs_enrichment flag), the integrity stage is
validate_enrichments_integrity. `labels` is the only field with
gold-truth semantics. A glossary in README/skill pins the vocabulary
(build pipeline / derive stage / canonical / labels / enrichments /
artifacts) plus the decision rule: consumed by a later stage as record
content → enrichment; the experiment's scored result → artifact.

**Notes:** LLM caches unaffected (keys hang off prompt/parser/model).
Old run snapshots carry the old `annotations` key — legacy expendable.
Pipelines whose YAML named the renamed stage types
(mark_needs_enrichment, validate_enrichments_integrity, enrichment_flag
overrides) change repro_hash — same accepted policy as before. Gotcha
for future renames: a blanket replace of "annotations" also hits
`from __future__ import annotations`; fix-forward was needed.

## 2026-07-04 — Naming pass: one classifier suffix, one metric vocabulary

**Problem:** Four spellings for "classifier" in stage types (_clf, _cls,
_classify, none), two vocabularies for the CV fold mean (classifier
stages: f1_mean; eval_predictions: f1), and three riddle names
(therefore, judge_aggregate vs its canonical instance judge_consensus,
ex_combined_binary).

**Solution:** All classifier stage types end in _clf with the backbone
first (bert_mlp_clf, bert_gated_clf, bert_text_clf, bert_text_cq_clf,
svm_clf, stacked_logreg_clf); bare metric key = fold mean everywhere
(the executor's {metric}_mean→{metric} table alias was removed);
therefore → therefore_components; judge_aggregate → judge_consensus
(type now equals the canonical instance name); ex_combined_binary →
explanation_combined_binary. table_method values were deliberately NOT
renamed (paper-table continuity).

**Notes:** Pipelines naming renamed stage types change repro_hash
(accepted). Old pipeline_summary files keep f1_mean in raw stage
metrics; their table_summary was computed at run time and stays
comparable. Kept on purpose: prompt_judge (term of art despite the
zs-judge collision), bayes/hungarian generate/compute pair pattern,
mark_cq_pilot.

## 2026-07-04 — Backend-aware model canon: identity = only what can affect output

**Problem:** The repro_hash canon took the flat ModelConfig wholesale
(minus description). Fields a backend ignores therefore changed run
identity: editing the (unused) transformer knobs on an LLM run, or
temperature on a CLI-backend run, made an identical run look new —
has_equivalent missed, compute was wasted.

**Solution:** Two-pronged. (a) Source cleanup: the dead ModelConfig
fields (num_labels, max_length, inference_batch_size, fewshot — zero
consumers after the TransformerModel removal) were deleted, so they no
longer exist in config_used.yaml either. (b) _model_canon in
repro_hash: description always volatile; temperature additionally
volatile for claude_cli/codex_cli (no CLI flag → cannot affect output).

**Notes:** The layer rule this encodes: YAML input = only deviations
(rule 6); config_used.yaml = FULL snapshot (forensic; "actually used"
is runtime-dependent and must not shape the record); repro_hash canon
= only output-affecting parameters. Canon change flips all repro
hashes once more (same week, same accepted policy) — after this the
canon is semantically tight: no known false sensitivity remains.

**Update (2026-07-12):** The which-backend-ignores-what knowledge moved
out of repro_hash into `BACKEND_CAPABILITIES` next to the backend
registry (`stagekit/models/backends/factory.py`), with an import-time
completeness check — registering a backend without declaring its
capabilities fails loud. `_model_canon` queries
`backend_has_temperature()`; unknown/legacy backend names default to
True (temperature stays in the hash), so stored hashes are unchanged.

## 2026-07-05 — Preflight/DAG visualization reuses the executor constructor

**Problem:** `run_experiment --dry-run` needs the executor's full
build-time validation (topo sort/cycles, enrichment + artifact
contracts, preloaded stages, preload hash/schema checks) without
executing stages. Reimplementing those checks in a preflight module
would drift from the executor; but `PipelineExecutor.__init__`
requires a RunManager, which creates run directories.

**Solution:** `stagekit/pipeline/preflight.py` constructs the REAL
`PipelineExecutor` with a `_NullRunManager` stub — construction runs
every validation, `run()` is never called, so the stub only needs
`load_pipeline_data() -> None`. Errors are collected into a
`PreflightReport` (not raised) so a multi-combo sweep shows every
problem at once, mirroring `ConfigValidationError`'s philosophy.
Renderers (`render_dag_ascii`, `render_dag_mermaid`) draw from the
resolved stage configs, so a broken DAG is still inspectable.

**Notes:** If `PipelineExecutor.__init__` ever grows a second
RunManager call, `_NullRunManager` fails loud (AttributeError)
rather than silently diverging. The dry run prints the DAG once per
distinct pipeline; per-combo lines show would-run/would-skip
(RunIndex `has_equivalent`) — a cheap pre-compute audit. Each DAG node
carries a one-line summary read from the stage class's OWN docstring
(`cls.__doc__`, not `inspect.getdoc` — so a stage without its own
docstring shows nothing, never the `PipelineStage` base text).

## 2026-07-12 — Closed config_override to declared keys (typo protection)

**Problem:** `required_config` only covered mandatory keys; every
optional knob was read via `co.get(key, default)` (correct per the
"defaults live in stage code" rule). Consequence: a typo in YAML
(`fold: 10` instead of `folds: 10`) validated fine and the stage
silently ran with the default — the worst failure mode for a research
pipeline, wrong-but-plausible results.

**Solution:** New class var `optional_config` on `PipelineStage`;
`allowed_config_keys()` unions required+optional over the MRO (so
`SVMClassifyStage` inherits `folds`/`seed`/`split_artifact` from
`KFoldClassifierStage` and declares only its own additions). The base
`check_stage_config` rejects any config_override key outside that set —
every override already called `super()`, so the check reaches all
stages. All 126 stage instances across the pipeline YAMLs were swept to
build the declarations; the sweep is exact, not guessed.

**Notes:** `table_method` is declared on the BASE class: it is legal on
any stage because `baseline_export._get_method_name` scans every
stage's config_override for it (and YAML sets it on `fallacies_binary`
stages, not just eval). When a stage grows a new `co.get(...)` key, the
declaration is part of the change — the unknown-key error message lists
the declared keys, so a missing declaration fails loud in preflight.

## 2026-07-23 — Stage-level resume removed: the LLM cache IS the resume mechanism

**Problem:** The executor carried a second resume path next to the LLM
cache: per-stage `_data_snapshot.jsonl` (full dataset copy after every
stage), `_skipped` markers, `pipeline.resume_from`, and `_apply_resume`
restore/clear logic. Cost: ~50% of `output/` (416 MB of 827 MB at
removal time) and the most intricate branch in the executor — for a
feature no experiment config ever set.

**Solution:** Deleted wholesale (executor, ArtifactManager,
RunManager delegates, `resume_from` config field + validation,
tests). An interrupted run is simply restarted: the persistent,
content-addressed `LLMCache` (key = prompt text + parser + template
hash) replays every already-answered call in seconds, and non-LLM stages
are seeded/deterministic. The only re-paid cost is classifier training
(minutes). `_completed` markers stay — they document which stages
finished, which is all one needs to decide whether to re-run.

**Notes:** Don't reintroduce snapshot-based resume for wall-clock
reasons; the cache already provides it content-addressed (survives
directory renames and new run dirs, which the snapshot path never did).
`resume_from` stays in `_VOLATILE_PIPELINE` (repro_hash) so stored
configs of old runs hash identically to new ones.

## 2026-07-23 — extensions/: parked subsystems are decommissioned, not half-alive

**Problem:** Idea-stage code (prompt-refinement subsystem, train_bert,
five never-built canonical builders) lived fully wired into the running
system — registry entries, configs, tests — although nothing used it.
Every audit had to re-discover that it was optional.

**Solution:** `extensions/` at the repo root, hard decommission: code
moved out of `src/`, registry entries (stage/parser/prompt maps) and
configs removed, tests moved alongside, excluded from ruff/pyright/
pytest. Reviving one is a deliberate checklist step (extensions/
README.md), not an accident. The generated-prompts infrastructure
(register_generated_prompts) stayed in src/prompts/catalog.py because
generated versions are drop-in templates for normal experiments.

**Notes:** Don't park things half-way (code in extensions/ but registry
entry kept): either it's wired in and tested, or it's out. mafalda the
*dataset config* stays active — only its unused canonical *builder*
moved.

## 2026-09-01 — vLLM backend: the reasoning trace must be separated by the server

**Problem:** vLLM speaks the OpenAI protocol, so "just point the openai
backend at it" looks like the whole job. It is not. Started without
`--reasoning-parser`, vLLM returns a reasoning model's thinking trace inside
`content` instead of in its own field, and nothing about the response looks
wrong. thesis-nandan measured what that costs there: on 2640 stored rows, 0
INVALID and 6.6 % of labels flipped one-directionally.

**That number does not transfer to this repo, and the first version of this
entry wrongly claimed it did** ("the parser takes the FIRST usable
candidate"). It does not: `parse_json_lenient` scans lines BOTTOM-UP for the
last one starting with `{` (stagekit/parsers/base.py). A trace followed by a
complete answer therefore parses correctly. Checked live against
`deepseek-r1_7b_vllm`, four items, re-parsing each answer as
`thinking + "\n" + content`: 0 of 4 flipped.

The real exposure here is the token budget, and it is silent. A deepseek-r1
trace runs ~1500 characters (~400 tokens). With `max_tokens` sized for the
answer alone, an unseparated trace spends the budget before the answer
starts — and what arrives is a deliberation containing a draft verdict,
which the parser dutifully reports. Same for an afterthought AFTER the
answer, since bottom-up means the last object wins. The four shapes and
which of them are silent are pinned in
`tests/test_parser_trace_robustness.py`.

**Solution:** `Answer(raw, thinking)` as the backend return type, so no
backend can hand a trace to a parser by accident; `reasoning: true|false`
declared per model; and `BackendContractError` when a `reasoning: true`
model returns content with no `reasoning_content`/`reasoning` field — the
exact shape of an unconfigured server.

Note what the guard is NOT for: it does not rescue a parser that would
otherwise be confused, because usually it would not be. It refuses a server
that is not the instrument the config declares — and that refusal is what
keeps the budget failure, which IS silent, from happening at all.

**Notes:** Four things that are not obvious and cost the most thought.

(a) *The guard had to be made fatal, or it would have been the most
expensive silent failure in the system.* `_classify_backend_error` matches
on message strings; a plain `RuntimeError` matches nothing, so the retry
loop would have hammered a permanently misconfigured server 10 times per
item. Hence the typed exception, an `isinstance` check before the string
matching, and `_FATAL_ERROR_TYPES` next to `auth`/`invalid_request`. A vLLM
HTTP 400 (over-long prompt) is already fatal via the "400" branch.

(b) *Polarity of the new capability defaults.* `_model_canon` is a denylist,
so `max_tokens`/`reasoning` on ModelConfig would have entered every run's
identity and flipped all stored repro hashes — and re-introduced exactly the
false sensitivity the 2026-07-04 canon change removed. Generalising
`BACKEND_CAPABILITIES` from `has_temperature` to declared `identity_fields`
BEFORE adding the fields avoids that. The defaults for an unknown backend
differ on purpose: `temperature` defaults True (stored configs carry it and
must keep hashing as before), the two new fields default False (no stored
config has them, so claiming them would invent identity). Verified, not
assumed: all 132 stored `config_used.yaml` re-hash identically to the
pre-change checkout, and `TestBackendAwareModelCanon.GOLDEN_OPENAI_02`
freezes one such hash as a literal.

(c) *Same rule for the cache key.* `backend_params` is omitted from the
payload when empty, so the four existing backends keep byte-identical keys
and the stored cache stays warm. It is filled from the same
`BACKEND_CAPABILITIES` declaration as the hash, so "different key" and
"different run identity" cannot drift apart.

(d) *`endpoint` is deliberately NOT a ModelConfig field.* The node changes
with every SLURM job and cannot affect output; as a config field it would
have to be excluded from hash and cache key immediately. It comes from
`VLLM_ENDPOINT`, like `OLLAMA_SERVER`. What is identity-bearing is the
instrument digest, which the swap proxy reports per alias (vLLM exposes no
weights hash, so the launch command IS the identity) and which lands in
`model_usage.json` — an observation about the run, deliberately outside
`config_used.yaml`.

Also recorded because they are asymmetries someone will trip over: a cache
HIT carries neither `raw` nor `thinking` (the trace is stored, but reachable
only via `get_entry`); the digest is re-fetched per stage rather than cached,
because the proxy re-reads its model table per request and a stale
instrument id is worse than none; and `--dry-run` does not build models, so
a dead endpoint surfaces at the first LLM stage, not at preflight.

**Open:** two measurements from thesis-nandan's plan apply here unchanged and
have not run. (A) Anchor: is vLLM-X the same instrument as ollama-X? Kill
below 95 % label agreement. (B) Concurrency invariance at temperature 0;
`VllmBackend.max_safe_workers` is pinned at 8 until it says otherwise. Until
A has run, a vLLM number and an ollama number of "the same" model are two
instruments.

**Null condition, run 2026-09-01 on a cluster node (vLLM 0.28.0).** The guard was
tested against a real misconfigured server, not only in unit tests: alias
`deepseek-r1_7b_vllm_noparser` in `swap-models.toml` serves the identical
weights *without* `--reasoning-parser` and exists only to fail. Three results.

(1) The instrument digest distinguishes them — `dbf6d2c1cc28` vs
`c22c763b970c` for the same `served_model`. The launch command is the
identity, as claimed.

(2) Declared `reasoning: false`, the server returns 1228 characters of
`content` with `thinking` empty: the trace inline, terminated by a literal
`</think>` marker, then the JSON. Deliberately NOT used — a marker heuristic
in the parser would be a second, weaker source of truth about something the
server can state properly.

(3) Declared `reasoning: true`, the guard raises, and over the full call path
with `max_retries=5` the backend is called exactly **once**. That was the
point of the typed exception.

**One bug surfaced by (3), fixed here:** the retry loop reported
`attempts=max_retries + 1` unconditionally, so a run stopping after one
attempt logged "attempts=6". It predates this work (`auth`/`invalid_request`
were already fatal) but was rare enough not to be noticed; `backend_contract`
makes it routine. A failure statistic that overstates itself is the silent
kind, so `_execute_llm_call` now counts what it did and the log says
"attempts=1 of 6 allowed".

## 2026-09-01 — Retry loop: three failures that report the wrong story

**Problem:** Writing the retry-loop tests (`tests/test_llm_retry_loop.py`)
turned up three defects in `stagekit/models/base.py`, all of the same family as
the `attempts` overcount above: the loop does something reasonable and then
reports it as something else.

(1) *A rate limit read as a bad request.* `_classify_backend_error` tested
`"400" in msg`. A real OpenAI 429 body is "Rate limit reached ... Limit:
40000 / min. Current: 41000 / min." — `"40000"` contains `"400"`, so a
retryable error was classified `invalid_request`, which is fatal, and the
item was abandoned on the first attempt. The same substring test made
"Limit: 500 / min" a `server_error`.

(2) *`error` and `error_type` from different attempts.* `last_error =
ar.error or last_error` kept the previous message when an attempt had none,
while `last_error_type` was overwritten regardless. A parse failure
following a backend error therefore reported `error="boom"` under
`error_type="parse_failure"`.

(3) *A fatal abort filed as retry exhaustion.* The `break` on a fatal error
ran into the same `_record_final_failure` as an exhausted budget, so the
item landed in `retry_exhausted_items`, `errors["<backend>:retries_exhausted"]`
rose, and `executor.py` warned "N failed items after retries" for items
attempted exactly once.

**Solution:** (1) status codes are matched as whole tokens
(`_STATUS_CODE_RE`, `\b\d{3}\b`) and the unambiguous `rate limit` phrase is
tested first; (2) message and type are assigned together, so both describe
the last attempt that had either; (3) `_record_final_failure` takes
`retries_exhausted`, and a fatal abort records a failed item but no retry
statistic, with the reason in the log line.

**Notes:** The classification stays best-effort string matching — a 429 body
that quotes "Limit: 400 / min" is still ambiguous, and only the leading
`rate limit` check saves it. Anything sharper wants the status code from the
backend rather than from `str(exc)`.


## 2026-09-02 — Repo-Audit: two cache contracts deliberately flipped

**Problem:** The full-repo audit (docs/audit-2026-09-02.md) found two cache
behaviours that looked like robustness but were silent-loss machines, plus a
batch of smaller fixes (A1-A7, B1, B3-B5 in the report). Two of the fixes
change contracts that tests had frozen, so the WHY belongs here.

**(1) Corrupt cache files now raise instead of "recovering".**
`_read_shard`/`_store_prompt` used to read a corrupt JSON file as empty; the
next `get_or_set` then REPLACED the file — permanent loss of every entry in
the shard, with the merge/run reporting success. Now they raise
`CacheCorruptionError`, `_execute_llm_call` re-raises it past the generic
handler (retrying cannot fix a broken file, and classification would have
retried it 10x per item), and the batch aborts fail-fast. The one tolerant
reader left is `get_entry` (debug/audit): read-only, cannot trigger the
overwrite, warns instead. `TestShardResilience` encoded the old behaviour as
correct and was rewritten — the "resilience" WAS the bug.

**(2) temperature enters the cache key only when the backend consumes it.**
`repro_hash._model_canon` already dropped temperature for `claude_cli`/
`codex_cli`; the cache key carried it unconditionally, so two CLI runs with
the SAME repro_hash had disjoint keys and paid every call twice — exactly
the key/identity drift the `_backend_params` docstring claimed impossible.
`make_key` now omits a `temperature=None`, fed from the same
`backend_consumes()` as the hash. Keys of consuming backends (openai/
ollama/vllm) stay byte-identical (golden test); CLI keys re-key once —
deliberate, legacy data is expendable, and CLI entries are few.

**Notes:** `required_config` now unions over the MRO
(`required_config_keys()`, same rule as `allowed_config_keys()`); the six
canonical builders lost their defensive duplicate `output_path`
declarations. The batch-wide `as_completed(timeout=600)` is gone — every
call is already bounded by its backend timeout plus retry budget, and a
10-minute cap over ALL futures aborted any batch whose legitimate total
runtime exceeded it, discarding finished results. Failed pipeline runs now
write `pipeline_status.json` (status="failed" + error) and a manifest via
try/except in `run_pipeline`, which is what makes `include_failed` in the
RunIndex mean something.


## 2026-09-15 — Backend architecture: why five backends, one interface

**Problem:** The pipeline supports models on heterogeneous infrastructure
(hosted APIs, local ollama, university vLLM cluster, local CLI agents),
but the cache key, repro hash, and retry logic must not leak backend
specifics upward.

**Solution:** `LLMBackend` (backends_base.py) defines a single contract:
`generate(prompt) -> Answer(raw, thinking)`. Five implementations
register via `@register_backend`:

| Backend | Transport | Identity fields | Notes |
|---------|-----------|-----------------|-------|
| `openai` | OpenAI SDK | temperature | Reasoning trace via `reasoning_content` when present |
| `ollama` | ollama SDK | temperature | Round-robin across `OLLAMA_SERVER` comma list; auto-pull |
| `vllm` | HTTP/requests | temperature, max_tokens, reasoning | Swap-proxy digest; `BackendContractError` on missing `--reasoning-parser` |
| `claude_cli` | subprocess | (none) | `claude -p` in sandbox, JSON wrapper |
| `codex_cli` | subprocess | (none) | `codex exec` read-only, raw stdout |

The key design decisions:

- **Backend = cache partition**: the backend name is part of the LLM
  cache key, because the same model name served by different backends is
  a different instrument (quantization, chat template).
- **Identity fields are declared, not assumed**: `BACKEND_CAPABILITIES`
  in factory.py lists which `ModelConfig` fields each backend passes to
  the model. Fields a backend ignores cannot affect output and are
  stripped from the repro hash and cache key. New backends must declare
  capabilities (enforced at import time).
- **Timeouts are env vars, not config**: they decide whether a call
  fails, not what the model answers — node properties, not run identity.
  `LLMBackend.timeout_from_env()` is the single mechanism.
- **Answer splits trace from answer**: `Answer(raw, thinking)` keeps
  the reasoning trace separate so parsers never see it. Without this,
  a reasoning trace merged into content silently flips labels (see
  2026-09-01 entry).

**Notes:** The `BackendContractError` path (vllm without
`--reasoning-parser`) is the one misconfiguration that looks like
correct output — content present, no error, but the trace consumed the
token budget and left a draft verdict as the last JSON. The backend
detects it (content without reasoning_content when reasoning=true) and
raises immediately instead of letting it through.


## 2026-09-02 — supports_temperature: why the hash machinery stays untouched

**Problem:** `NO_TEMPERATURE_MODELS` in the openai backend hardcoded model
IDs (drift: a new snapshot silently escaped the guard) and silently forced
temperature=1 while the configured temperature still entered hash and cache
key. The obvious fix — strip temperature from the canon when a model
declares `supports_temperature: false` — is WRONG: stored gpt-5 runs carry
`temperature: 1` in their canon, so stripping would re-key every one of
them.

**Solution:** The flag moved to the model YAML
(`ModelConfig.supports_temperature`), and the drift is closed by
VALIDATION instead of hashing: a config with `supports_temperature: false`
and `temperature != 1` is rejected at preflight (and the backend raises on
direct construction). For every config that can run, temperature is 1, so
no false sensitivity can arise — and hashes/cache keys stay byte-stable.
`supports_temperature` itself is registered in CONDITIONAL_FIELD_DEFAULTS
with default False (never identity), so declaring it cannot re-key a run.

**Notes:** Also from this round: exports read the run's recorded
repro_hash from `config_used.yaml` (`stored_repro_hash`) instead of
recomputing on the post-run cfg; path-key preflight checks are declared on
the stage class (`input_path_config_keys`/`output_path_config_keys`,
MRO-union) after the central list in validation.py had already drifted
(`taxonomy_path` was unchecked); OLLAMA_TIMEOUT_S is an env var like
OLLAMA_SERVER because a timeout is a node property, not run identity.

## 2026-10-08 — published/: pins instead of copies

**Problem:** `published/` copied every run's cross-run artifacts after every
run (110 MB), decided "which run is the input" by file permissions
(`chmod` lock, `PermissionError`, three "published p1 replaced" incidents),
and its `source_run_dir` backlink was a path, not a key. Two export
modules, two repair scripts, no explicit act of choosing a run. Decision
note: `docs/published-pins-statt-kopien.md`.

**Solution:** `published/pins.yaml` (git) maps a coordinate — the address
consumer YAMLs interpolate, `counterfactual/<data>/<model>/<mode>/p<n>` or
`baselines/<data>/<model>/<method>` — to one run: `run_dir`, `run_id`,
`repro_hash`, SHA-256 per artifact. `preload_artifacts` values are
`pin:<coordinate>`; the executor, config validation and the multi-mode
loader (`load_mode_artifacts`) resolve through `stagekit/run/pins.py` and read
the artifact from the run itself. Pinning is `python -m scripts.pin
<run_dir> <coordinate>` after the run; the runner only prints that command
(`suggest_coordinate`). A missing pin lists candidate runs, never takes
the newest. `RunRecord.pinned` (catalog v3, pins.yaml mtime in the cache
key) marks runs that must survive a cleanup; `clean.sh` warns.
Migration (`scripts/migrate_published_to_pins.py`) verified all 44
exports against their runs before writing a pin — byte-equal or
JSON-equal, derived files via the derivation — and `pin --list` shows
44 OK.

**Notes (deviations from the decision note, found while building):**
- The note's "the copy does no work the run does not" was wrong twice.
  `predictions_features.json` never was a stage artifact (computed at
  export time), and 18 of 26 counterfactual exports carry a
  `cf_likelihoods.json` their 2026-05 run never wrote (computed post hoc
  by the deleted `reexport_all.py`). First attempt: a `derive:` field in
  the pin. Rejected the same day — a pin would then compute with whatever
  code is current, unhashed (the pin hashes the source, not the result),
  and label the result as the producing run's output. Now the CONSUMER
  declares it: `pipeline.derived_artifacts` (`derived.<field>: {from:
  <preloaded key>, derive: <name>}`, `stagekit/stages/derived_artifacts.py`).
  The executor materialises the keys after preloading, the stacked SVM
  derives per pinned mode (`load_mode_artifacts`), the run's
  `run_inputs.yaml` records the declaration — provenance is the consumer
  run's, pins stay pointers. The pseudo-stage `derived` keeps the field as
  the feature-name prefix, so `cf_likelihoods_…` names are unchanged. Not
  a derive stage because the stacked SVM reads three pinned runs, not
  stage artifacts. The migration verifies computed copies against the
  derivation and does not pin them.
- Baseline predictions are pinned under role keys (`baseline.predictions`,
  `llm_baseline.predictions_features`): consumers address a baseline by
  the method axis and must not know the producing stage. The 2026-05
  ex_combined runs named their stage `ex_combined_binary` (type since
  renamed); the name is in `BASELINE_PREDICTION_STAGES` for them.
- 38 of the 44 pinned runs predate the `artifacts/` → `stage_outputs/`
  rename. `ArtifactManager.for_run` now reads the directory from the run's
  own `run_manifest.json` (`paths.stage_outputs_dir` / `artifacts_dir`),
  so every reader of old runs works, not only pins.
- File paths in `preload_artifacts` are no longer accepted (`pin:` only);
  `preload_expect` keys are `run_id`, `repro_hash`, `artifact_hashes`
  ({artifact_key: sha256}); `metadata.json` is gone, with it
  `_validate_metadata_hashes`. `verify_recorded_hashes` stays the one hash
  check.
- Pre-existing bug surfaced by the fusion preflight: the per-mode preload
  check derived a mode path for EVERY preload key, including the
  global-scoped baseline (`feature_source_scopes: global`), whose address
  has no mode segment — the fusion experiment could not pass preflight.
  `global_feature_keys` (feature_utils) now excludes them, the same split
  the runtime makes.
- Not built: `pin --copy` (open question 2 of the note; nothing has been
  deleted from `output/` yet) and the `p<n>` segment for baseline
  coordinates (open question 1). `scripts/migrate_variant_names.py`
  (one-time, 2026-07) deleted with its test instead of rewired; seven
  pipeline YAMLs lost an `artifact_root` nobody read.
- `repro_hash`: the consumer pipelines' hash includes the preload values,
  so switching them from paths to `pin:` coordinates re-keys the four
  consumer pipelines — their old runs are no longer "equivalent" for
  `has_equivalent` (the paper reads `paper/runs.json`, not the newest
  run, so nothing moves). `derived_artifacts` hashes as absent when empty
  (`_EMPTY_IS_ABSENT_PIPELINE`), so every other stored run keeps its
  identity. Known gap, same as before with paths: the coordinate is in
  the hash, the pinned run's hashes are not — re-pinning changes the
  input without changing the consumer's `repro_hash`; the consumer run's
  `preload_provenance.json` is the record.
- The copies under `published/counterfactual/` and `published/baselines/`
  are deletable once this is committed (`chmod -R u+w` first; the files
  are read-only).

## 2026-10-13 — Rule 6 enforced in the pipeline YAMLs: 77 default-restating lines removed, and what that does to repro_hash

**Problem:** `config_override` blocks restated the stage-code default in 77
places (`folds: 5`/`seed: 42` in nine pipelines, `kernel: rbf`/`C: 1.0` on
all 23 SVM stages, `drop_duplicates`, `cleanup_result`, `output_field`,
`enrichment_version`, `canonical_version`, …). The September audit's B8 fix
had only cleaned the four `bert-text*.yaml`. A restated default masks a
later change of the code default: the YAML keeps the old value and the run
stays green.

**Solution:** Line-based strip with a before/after YAML comparison proving
that the only change is the removal of the listed (stage_type, key, value)
triples; emptied `config_override:` blocks dropped. Script in the session
scratchpad, not kept — the inventory it was built on is one `yaml` walk
over `config/pipeline/**` against the `.get("key", default)` calls of the
stage module.

**Notes:** `repro_hash` canonicalises the full stage definitions
(`stagekit/analysis/repro_hash.py:_build_canonical`), so every touched pipeline
now hashes differently from its earlier runs. Pins are unaffected (they
point at `run_dir`), the paper tables come from pins, but a RunIndex
comparison across the change will see the next run of these pipelines as
new. No stage default is declared declaratively, so a preflight that
rejects `value == default` would need a second home for the defaults —
not built; the rule stays a review item.
