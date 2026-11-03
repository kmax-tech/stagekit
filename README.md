# stagekit

A framework for running experiments as configured pipeline stages,
reproducible by hash: pipeline stages and their DAG executor, registries
a topic adds to (`stagekit/run/topics.py`), parsing, the repro hash, the
run index and pins.

It was split off from fallacy-detection on 2026-11-03, its first topic.
This repository starts with one commit; the history up to the split
stays in fallacy-detection, where the core left in the commit "stagekit
leaves: the core is removed here and installed editable" of that day.

A project using stagekit has `config/config.yaml` (it names its topics)
and runs from its root, or sets `STAGEKIT_PROJECT_ROOT`
(`stagekit/config/paths.py`). The interface is provisional (0.x) until a
second topic runs on it unchanged (plan, step 6).

Architecture reference: `.claude/commands/experiment.md`. Code style:
`docs/coding-style.md`. Why hard problems were solved as they were:
`NOTES.md`.

```bash
pip install -e .            # or -e .[bert,tracking]
python -m pytest tests/ -q
```
