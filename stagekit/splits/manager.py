from pathlib import Path
from typing import Dict
from stagekit.splits.registry import create_split_strategy
from stagekit.splits.config import load_split
from stagekit.utils.serialization import save_json


class SplitManager:
    def __init__(self, split_dir: Path):
        self.split_dir = split_dir

    def load(self, split_id: str, dataset=None) -> Dict:
        path = self.split_dir / f"{split_id}.json"
        # Widen to a plain dict: this method accesses split keys by
        # computed name, which a TypedDict cannot type.
        split: dict = dict(load_split(path))

        # If no dataset provided, return raw split
        if dataset is None:
            return split

        # Build ID → index map
        id2idx = {id_: i for i, id_ in enumerate(dataset.ids)}

        resolved = {}

        all_split_ids = set()

        for key in ("train", "dev", "test", "val"):
            ids_key = f"{key}_ids"
            idx_key = f"{key}_idx"

            if ids_key in split:
                all_split_ids.update(split[ids_key])
                resolved[idx_key] = [
                    id2idx[id_] for id_ in split[ids_key] if id_ in id2idx
                ]
            else:
                resolved[idx_key] = split.get(idx_key, [])

        missing = all_split_ids - set(id2idx)
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(all_split_ids)} IDs from split "
                f"'{split_id}' not found in dataset "
                f"(first: {sorted(missing)[:3]}). "
                f"The split file does not match the loaded dataset."
            )

        return resolved

    def create_and_save(self, split_id: str, strategy_name: str, dataset, **kwargs):
        self.split_dir.mkdir(parents=True, exist_ok=True)
        strategy = create_split_strategy(strategy_name, **kwargs)
        split = strategy.apply(dataset)

        payload = {
            "split_id": split_id,
            "strategy": strategy_name,
            **kwargs,
            **split,
        }

        path = self.split_dir / f"{split_id}.json"
        if path.exists():
            raise FileExistsError(f"Split already exists: {path}")

        save_json(payload, path)
        return payload
