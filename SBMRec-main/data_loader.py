# -*- coding: utf-8 -*-
"""Data preparation for multimodal multi-behavior sequential recommendation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


USER_COLUMNS = ("user_id", "user", "uid", "userid")
ITEM_COLUMNS = ("item_id", "item", "iid", "video_id", "live_id", "streamer_id")
TIME_COLUMNS = ("timestamp", "time", "ts", "date", "datetime", "event_time")
BEHAVIOR_COLUMNS = ("behavior", "behavior_type", "event_type", "action", "type")

CLICK_ALIASES = {"click", "clicked"}
FAVOR_ALIASES = {"favor", "favorite", "favour", "favourite", "fav", "collect"}

CACHE_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_behavior_values(values: Sequence[str]) -> Set[str]:
    result: Set[str] = set()
    for value in values:
        for part in str(value).split(","):
            normalized = part.strip().lower()
            if normalized:
                result.add(normalized)
    return result


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower_to_original = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return str(lower_to_original[candidate])
    return None


def _infer_behavior_from_filename(
    path: Path, click_values: Set[str], favor_values: Set[str]
) -> Optional[str]:
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    matches = {
        alias for alias in click_values | favor_values
        if not alias.isdigit() and (alias in tokens or f"{alias}s" in tokens)
    }
    if matches & click_values and matches & favor_values:
        raise ValueError(f"Ambiguous behavior in filename {path.name}; add a behavior column.")
    return sorted(matches)[0] if matches else None


def _sort_chronologically(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by user, timestamp, and original row order (stable)."""
    if "timestamp" not in frame.columns:
        return frame.sort_values(["user_id", "_row_order"], kind="stable")

    timestamp = frame["timestamp"]
    if timestamp.isna().any() or timestamp.astype(str).str.strip().eq("").any():
        raise ValueError("Every retained interaction must have a valid timestamp.")
    numeric = pd.to_numeric(timestamp, errors="coerce")
    if numeric.notna().all() and np.isfinite(numeric.to_numpy()).all():
        frame = frame.assign(_sort_time=numeric)
    else:
        if numeric.notna().any():
            raise ValueError("Timestamps mix numeric and non-numeric values.")
        parsed = pd.to_datetime(timestamp, errors="coerce", utc=True, format="mixed")
        if parsed.isna().any():
            raise ValueError("Some timestamps cannot be parsed.")
        frame = frame.assign(_sort_time=parsed)
    return frame.sort_values(
        ["user_id", "_sort_time", "_row_order"], kind="stable"
    ).drop(columns=["_sort_time"])


def _iterative_filter(
    frame: pd.DataFrame,
    min_user_interactions: int,
    min_item_freq: int,
    min_favor_interactions: int,
) -> pd.DataFrame:
    filtered = frame
    while True:
        before = len(filtered)
        if min_item_freq > 1:
            item_counts = filtered["item_id"].value_counts()
            filtered = filtered[filtered["item_id"].isin(item_counts[item_counts >= min_item_freq].index)]
        if min_user_interactions > 1:
            user_counts = filtered["user_id"].value_counts()
            filtered = filtered[filtered["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)]
        favor_counts = filtered.groupby("user_id")["is_favor"].sum()
        filtered = filtered[filtered["user_id"].isin(favor_counts[favor_counts >= min_favor_interactions].index)]
        if len(filtered) == before:
            break
    return filtered.copy()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@dataclass
class DatasetBundle:
    train: "MultiBehaviorSequenceDataset"
    validation: "MultiBehaviorSequenceDataset"
    test: "MultiBehaviorSequenceDataset"
    num_items: int                   # catalog size including the padding index 0
    raw_item_ids: List[str]          # raw_item_ids[i] is the raw id of item index i
    item2idx: Dict[str, int]
    user_seen_items: List[Set[int]]  # all recorded items per user (evaluation candidates)
    train_seen_items: List[Set[int]] # items before the validation target (training negatives)
    raw_user_ids: List[str]


class MultiBehaviorSequenceDataset(Dataset):

    def __init__(
        self,
        user_items: Sequence[np.ndarray],
        user_is_favor: Sequence[np.ndarray],
        targets: Sequence[Tuple[int, int]],
        seq_len: int,
        split: str,
    ) -> None:
        if seq_len <= 0 or seq_len % 2:
            raise ValueError("seq_len must be a positive even number for the Haar DWT.")
        self.user_items = user_items
        self.user_is_favor = user_is_favor
        self.targets = list(targets)
        self.seq_len = seq_len
        self.split = split

    def __len__(self) -> int:
        return len(self.targets)

    def _pad(self, values: np.ndarray) -> torch.Tensor:
        values = values[-self.seq_len:]
        padded = np.zeros(self.seq_len, dtype=np.int64)
        padded[: len(values)] = values
        return torch.from_numpy(padded)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        user_index, target_position = self.targets[index]
        items = self.user_items[user_index]
        is_favor = self.user_is_favor[user_index]

        history_items = items[:target_position]
        history_is_favor = is_favor[:target_position]
        return {
            "click_seq": self._pad(history_items[~history_is_favor]),
            "favor_seq": self._pad(history_items[history_is_favor]),
            "labels": torch.tensor(int(items[target_position]), dtype=torch.long),
            "user_index": torch.tensor(user_index, dtype=torch.long),
        }


def _make_bundle(
    user_items: Sequence[np.ndarray],
    user_is_favor: Sequence[np.ndarray],
    train_targets: Sequence[Tuple[int, int]],
    validation_targets: Sequence[Tuple[int, int]],
    test_targets: Sequence[Tuple[int, int]],
    seq_len: int,
    num_items: int,
    raw_item_ids: List[str],
    item2idx: Dict[str, int],
    user_seen_items: List[Set[int]],
    raw_user_ids: List[str],
) -> DatasetBundle:
    # Training negatives exclude only items that appear before the validation
    # target, so that no validation or test interaction is used during training.
    train_seen_items: List[Set[int]] = [set() for _ in user_items]
    for user_index, validation_position in validation_targets:
        train_seen_items[user_index] = set(
            int(item) for item in user_items[user_index][:validation_position]
        )
    return DatasetBundle(
        train=MultiBehaviorSequenceDataset(user_items, user_is_favor, train_targets, seq_len, "train"),
        validation=MultiBehaviorSequenceDataset(user_items, user_is_favor, validation_targets, seq_len, "validation"),
        test=MultiBehaviorSequenceDataset(user_items, user_is_favor, test_targets, seq_len, "test"),
        num_items=num_items,
        raw_item_ids=raw_item_ids,
        item2idx=item2idx,
        user_seen_items=user_seen_items,
        train_seen_items=train_seen_items,
        raw_user_ids=raw_user_ids,
    )

def _load_interactions(
    data_root: str,
    click_values: Set[str],
    favor_values: Set[str],
    assume_chronological: bool,
) -> pd.DataFrame:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {data_root}")

    frames = []
    row_offset = 0
    for path in csv_files:
        frame = pd.read_csv(path, dtype=str)  # keep raw identifiers unchanged
        user_column = _find_column(frame.columns, USER_COLUMNS)
        item_column = _find_column(frame.columns, ITEM_COLUMNS)
        time_column = _find_column(frame.columns, TIME_COLUMNS)
        behavior_column = _find_column(frame.columns, BEHAVIOR_COLUMNS)
        if user_column is None or item_column is None:
            print(f"[Warn] Skip {path.name}: cannot identify user/item columns.")
            continue
        if time_column is None and not (assume_chronological and len(csv_files) == 1):
            raise ValueError(
                f"{path.name} has no timestamp column; timestamps are required to "
                "order interactions across behaviors."
            )

        frame = frame.dropna(subset=[user_column, item_column])
        frame = frame[
            frame[user_column].str.strip().ne("") & frame[item_column].str.strip().ne("")
        ].reset_index(drop=True)

        selected = pd.DataFrame({
            "user_id": frame[user_column].str.strip(),
            "item_id": frame[item_column].str.strip(),
            "_row_order": np.arange(row_offset, row_offset + len(frame)),
        })
        row_offset += len(frame)
        if time_column is not None:
            selected["timestamp"] = frame[time_column]

        if behavior_column is not None:
            selected["behavior_raw"] = frame[behavior_column].str.strip().str.lower()
        else:
            inferred = _infer_behavior_from_filename(path, click_values, favor_values)
            if inferred is None:
                print(f"[Warn] Skip {path.name}: behavior cannot be determined.")
                continue
            selected["behavior_raw"] = inferred
        frames.append(selected)

    if not frames:
        raise RuntimeError("No valid interaction CSV could be parsed.")

    interactions = pd.concat(frames, ignore_index=True)
    interactions = interactions.dropna(subset=["user_id", "item_id", "behavior_raw"])
    interactions["behavior_raw"] = interactions["behavior_raw"].astype(str).str.strip().str.lower()
    known = interactions["behavior_raw"].isin(click_values | favor_values)
    if (~known).any():
        print(f"[Info] Drop {int((~known).sum())} interactions of other behavior types.")
        interactions = interactions[known]
    interactions["is_favor"] = interactions["behavior_raw"].isin(favor_values)
    return _sort_chronologically(interactions)


def _cache_key(csv_files: Sequence[Path], settings: Dict[str, object]) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        **settings,
        "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in csv_files],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def build_datasets(
    data_root: str,
    seq_len: int = 50,
    min_user_interactions: int = 5,
    min_favor_interactions: int = 3,
    min_item_freq: int = 1,
    click_behaviors: Sequence[str] = ("click",),
    favor_behaviors: Sequence[str] = ("favor",),
    use_cache: bool = True,
    assume_chronological: bool = False,
) -> DatasetBundle:
    if seq_len <= 0 or seq_len % 2:
        raise ValueError("seq_len must be a positive even number for the Haar DWT.")
    if min_user_interactions < 1 or min_item_freq < 1 or min_favor_interactions < 3:
        raise ValueError("Each user needs at least three favorites (train/validation/test).")

    click_values = parse_behavior_values(click_behaviors)
    favor_values = parse_behavior_values(favor_behaviors)
    if click_values & CLICK_ALIASES:
        click_values |= CLICK_ALIASES
    if favor_values & FAVOR_ALIASES:
        favor_values |= FAVOR_ALIASES
    if not click_values or not favor_values or click_values & favor_values:
        raise ValueError("Click and favorite behavior values must be nonempty and disjoint.")

    root = Path(data_root)
    csv_files = sorted(root.glob("*.csv"))
    key = _cache_key(csv_files, {
        "seq_len": seq_len,
        "min_user_interactions": min_user_interactions,
        "min_favor_interactions": min_favor_interactions,
        "min_item_freq": min_item_freq,
        "click_values": sorted(click_values),
        "favor_values": sorted(favor_values),
        "assume_chronological": assume_chronological,
    })
    cache_path = root / ".sbmrec_cache" / f"processed_{key}.pkl"
    if use_cache and cache_path.exists():
        print(f"[Info] Load processed data from {cache_path}")
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    interactions = _load_interactions(data_root, click_values, favor_values, assume_chronological)
    interactions = _iterative_filter(
        interactions, min_user_interactions, min_item_freq, min_favor_interactions
    )
    if interactions.empty:
        raise RuntimeError("No interaction remains after filtering.")

    raw_item_ids_no_pad = interactions["item_id"].drop_duplicates().tolist()
    item2idx = {str(raw_id): index + 1 for index, raw_id in enumerate(raw_item_ids_no_pad)}
    interactions["item_index"] = interactions["item_id"].map(item2idx).astype(np.int64)
    raw_item_ids = ["__PAD__"] + [str(item) for item in raw_item_ids_no_pad]

    user_items: List[np.ndarray] = []
    user_is_favor: List[np.ndarray] = []
    user_seen_items: List[Set[int]] = []
    raw_user_ids: List[str] = []
    train_targets: List[Tuple[int, int]] = []
    validation_targets: List[Tuple[int, int]] = []
    test_targets: List[Tuple[int, int]] = []

    for raw_user_id, user_frame in interactions.groupby("user_id", sort=False):
        items = user_frame["item_index"].to_numpy(dtype=np.int64, copy=True)
        is_favor = user_frame["is_favor"].to_numpy(dtype=bool, copy=True)
        favor_positions = np.flatnonzero(is_favor)
        user_index = len(user_items)

        raw_user_ids.append(str(raw_user_id))
        user_items.append(items)
        user_is_favor.append(is_favor)
        user_seen_items.append(set(int(item) for item in items.tolist()))

        for position in favor_positions[:-2]:
            if position > 0:
                train_targets.append((user_index, int(position)))
        validation_targets.append((user_index, int(favor_positions[-2])))
        test_targets.append((user_index, int(favor_positions[-1])))

    if not train_targets:
        raise RuntimeError("No training target was generated.")

    bundle = _make_bundle(
        user_items, user_is_favor, train_targets, validation_targets, test_targets,
        seq_len, len(raw_item_ids), raw_item_ids, item2idx, user_seen_items, raw_user_ids,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (cache_path.parent / f"item_mapping_{key}.json").open("w", encoding="utf-8") as handle:
            json.dump(item2idx, handle, ensure_ascii=False)
        print(f"[Info] Saved processed data to {cache_path}")
    return bundle


# ---------------------------------------------------------------------------
# Pre-extracted CLIP features
# ---------------------------------------------------------------------------
def load_item_features(path: str, raw_item_ids: Sequence[str]) -> torch.Tensor:
    """Return a [num_items, d_c] feature table aligned with the item indices."""

    file = Path(path)
    if file.suffix == ".npz":
        data = np.load(file, allow_pickle=True)
        mapping = dict(zip((str(i) for i in data["item_ids"]), data["features"]))
    else:
        if file.suffix in {".pt", ".pth"}:
            obj = torch.load(file, map_location="cpu")
        else:
            with file.open("rb") as handle:
                obj = pickle.load(handle)
        if isinstance(obj, dict) and "item_ids" in obj and "features" in obj:
            mapping = dict(zip((str(i) for i in obj["item_ids"]), obj["features"]))
        elif isinstance(obj, dict):
            mapping = {str(k): v for k, v in obj.items()}
        else:
            raise ValueError(f"Unsupported feature file content: {path}")

    missing = [raw for raw in raw_item_ids[1:] if raw not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} items have no feature in {path}, e.g. {missing[:5]}")
    first = np.asarray(next(iter(mapping.values())), dtype=np.float32).reshape(-1)
    table = np.zeros((len(raw_item_ids), first.shape[0]), dtype=np.float32)
    for index, raw in enumerate(raw_item_ids[1:], start=1):
        table[index] = np.asarray(mapping[raw], dtype=np.float32).reshape(-1)
    return torch.from_numpy(table)
