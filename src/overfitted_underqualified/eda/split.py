"""Canonical copies, diagnostic scene similarity and reproducible grouped splitting."""
import itertools
from sklearn.model_selection import GroupShuffleSplit
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .video import clip_sort_key
from .annotations import LABELS

# Канонические ID и таблица уникальных видео с сохранением всех событий.

def duplicate_mapping(inventory):
    mapping = inventory[["clip_id", "sha256"]].copy()
    mapping["canonical_clip_id"] = mapping["clip_id"]
    for digest, rows in mapping.loc[mapping["sha256"].ne("")].groupby("sha256"):
        mapping.loc[rows.index, "canonical_clip_id"] = min(rows["clip_id"], key=clip_sort_key)
    return mapping


def training_video_table(analysis):
    """One example per unique video, retaining all source IDs and event times."""
    columns = ["clip_id", "source_label", "video_file", "sha256", "duration_sec", "fps",
               "width", "height", "source_clip_ids", "event_times_sec"]
    rows = []
    for canonical, group in analysis.groupby("canonical_clip_id", sort=False):
        if group["source_label"].nunique() != 1:
            raise ValueError(f"Conflicting labels for identical video: {canonical}")
        representative = group.loc[group["clip_id"].eq(canonical)]
        row = (representative if not representative.empty else group).iloc[0]
        record = {c: row[c] for c in columns if c in row.index}
        record["clip_id"] = canonical
        record["source_clip_ids"] = json.dumps(sorted(group["clip_id"].tolist(), key=clip_sort_key))
        record["event_times_sec"] = json.dumps(sorted(set(group["timecode_sec"].dropna().astype(float))))
        rows.append(record)
    return pd.DataFrame(rows, columns=columns)

# Поиск визуально похожих сцен без использования меток.

def scene_candidates(thumbnails, threshold=0.12):
    signatures = {clip_id: np.median(np.stack(frames), axis=0).astype(float) / 255
                  for clip_id, frames in thumbnails.items() if frames}
    pairs = []
    for left, right in itertools.combinations(sorted(signatures), 2):
        distance = float(np.mean(np.abs(signatures[left] - signatures[right])))
        if distance <= threshold:
            pairs.append({"clip_a": left, "clip_b": right, "scene_distance": distance})
    return pd.DataFrame(pairs, columns=["clip_a", "clip_b", "scene_distance"]).sort_values("scene_distance")

# Группировка связанных видео и предложение разбиения без пересечения групп.

def build_groups(events, inventory, pairs, keep_together=False):
    ids = sorted(events["clip_id"].dropna().unique(), key=clip_sort_key)
    parent = {clip_id: clip_id for clip_id in ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        if left in parent and right in parent:
            parent[find(right)] = find(left)

    for _, rows in inventory.loc[inventory["sha256"].ne("")].groupby("sha256"):
        values = rows["clip_id"].tolist()
        for value in values[1:]:
            union(values[0], value)
    for row in pairs.itertuples(index=False):
        union(row.clip_a, row.clip_b)
    if keep_together and ids:
        for value in ids[1:]:
            union(ids[0], value)
    return pd.DataFrame([{"clip_id": clip_id, "group_id": find(clip_id)} for clip_id in ids],
                        columns=["clip_id", "group_id"])


def propose_split(events, groups, seed=42):
    data = events[["clip_id", "source_label"]].merge(groups, on="clip_id", validate="one_to_one")
    if set(data["source_label"]) != set(LABELS) or data["group_id"].nunique() < 2:
        return pd.DataFrame(), "Insufficient video groups or classes"
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.25, random_state=seed)
    best, best_score = None, np.inf
    for train, validation in splitter.split(data, groups=data["group_id"]):
        left, right = data.iloc[train], data.iloc[validation]
        if any(set(part["source_label"]) != set(LABELS) for part in (left, right)):
            continue
        prevalence = data["source_label"].eq("violation").mean()
        score = abs(len(right) / len(data) - 0.25) + sum(
            abs(part["source_label"].eq("violation").mean() - prevalence) for part in (left, right))
        if score < best_score:
            best_score, best = score, (train, validation)
    if best is None:
        return pd.DataFrame(), "Cannot keep both classes in each subset without splitting groups"
    data["split"] = "train"
    data.loc[data.index[best[1]], "split"] = "validation"
    assert data.groupby("group_id")["split"].nunique().max() == 1
    return data, "provisional within-room split; recording-session independence is unknown"


def preserve_split(proposed, path):
    """Validate against the committed assignment; never silently replace an existing split."""
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(existing.reset_index(drop=True), proposed.reset_index(drop=True),
                                      check_dtype=False, check_like=True)
    else:
        proposed.to_csv(path, index=False)


def prepare_split(analysis, inventory, training_videos, output_dir, seed=42):
    """Check canonical/source coverage and leakage, then validate the saved assignment."""
    # Для согласованного предварительного split группируем только точные копии.
    # Пары визуального сходства остаются диагностикой, а не независимыми сеансами.
    groups = build_groups(analysis, inventory, pd.DataFrame(columns=["clip_a", "clip_b"]))
    groups.to_csv(output_dir / "scene_groups.csv", index=False)
    split, split_status = propose_split(training_videos, groups, seed)
    split_path = output_dir / "split_proposal.csv"
    if not split.empty:
        split = split.merge(training_videos[["clip_id", "sha256", "source_clip_ids", "event_times_sec"]], on="clip_id", validate="one_to_one")
        split["status"] = "provisional"
        split["protocol"] = "within_room_unique_video"
        assert set(split["clip_id"]) == set(training_videos["clip_id"])
        assert not split["clip_id"].duplicated().any()
        assert split.groupby("sha256")["split"].nunique().max() == 1
        assert split.groupby("split")["source_label"].nunique().eq(2).all()
        source_assignments = analysis[["clip_id", "canonical_clip_id", "sha256"]].merge(
            split[["clip_id", "split"]].rename(columns={"clip_id": "canonical_clip_id"}),
            on="canonical_clip_id", validate="many_to_one")
        assert len(source_assignments) == len(analysis) and source_assignments["split"].notna().all()
        assert source_assignments.groupby("sha256")["split"].nunique().max() == 1
        preserve_split(split, split_path)
    else:
        raise ValueError(f"Разбиение недоступно: {split_status}; существующий CSV сохранён")

    return groups, split, split_status
