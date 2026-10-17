"""EDA report orchestration with explicit inputs and output directories."""
import hashlib
import os
import json
from pathlib import Path
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm
from .video import collect_videos, inspect_video, clip_sort_key
from .annotations import parse_timecode, validate_dataset
from .split import duplicate_mapping, training_video_table


def inspect_directory(root, processed_dir):
    explicit = os.environ.get("VIDEO_DIR") or os.environ.get("FORTEBANK_OPEN_DIR")

    if explicit:
        data_dir = Path(explicit).expanduser().resolve()
        if not data_dir.is_dir() or not collect_videos(data_dir):
            raise FileNotFoundError(f"В указанном каталоге нет видео: {data_dir}")
    else:
        candidates = [root / p for p in ("open", "data/open", "data/raw/open", "data/raw/videos", "data/raw")]
        data_dir = next((p for p in candidates if p.is_dir() and collect_videos(p)), None)
        if data_dir is None:
            raise FileNotFoundError("Видео не найдены. Добавьте их в open/ или задайте VIDEO_DIR.")

    print(f"Видео: {data_dir}")

    video_paths = collect_videos(data_dir)
    records, thumbnails = [], {}
    for path in tqdm(video_paths, desc="video checks"):
        record, sampled = inspect_video(path, data_dir)
        records.append(record)
        # Не выбираем случайный файл при одинаковом идентификаторе.
        thumbnails.setdefault(record["clip_id"], sampled)
    inventory = pd.DataFrame(records)
    inventory.to_csv(processed_dir / "video_inventory.csv", index=False)
    readable = inventory.loc[inventory["readable"]].copy()
    print(f"Файлов: {len(inventory)}, прошли выборочное чтение: {len(readable)}")
    display(inventory)
    display(inventory.loc[inventory["clip_id"].duplicated(False), ["clip_id", "video_rel"]])
    display(inventory.loc[inventory["sha256"].ne("") & inventory["sha256"].duplicated(False), ["clip_id", "video_rel", "sha256"]])
    return data_dir, inventory, readable, thumbnails


def prepare_annotations(events_path, inventory, processed_dir):
    if not events_path.is_file():
        raise FileNotFoundError(f"Нет исходной разметки: {events_path}")
    source_digest = hashlib.sha256(events_path.read_bytes()).hexdigest()
    events = pd.read_csv(events_path, dtype={"clip_id": "string", "timecode": "string"})
    required = {"clip_id", "timecode", "source_label", "observed_event", "observed_event_type", "candidate_violation_type"}
    if missing := required - set(events.columns):
        raise ValueError(f"В events.csv отсутствуют колонки: {sorted(missing)}")
    events["timecode_sec"] = events["timecode"].map(parse_timecode)
    validation = validate_dataset(events, inventory)
    notes = validation.loc[validation["severity"].eq("info")]
    notes.to_csv(processed_dir / "dataset_notes.csv", index=False)
    issues = validation.loc[~validation["severity"].eq("info")].copy()
    issues.to_csv(processed_dir / "validation_issues.csv", index=False)
    display(issues)
    blocked = set(issues.loc[issues["severity"].eq("error"), "clip_id"])
    eligible = events.loc[~events["clip_id"].astype(str).isin(blocked)].copy()
    technical = inventory.loc[~inventory["clip_id"].duplicated(False)].rename(columns={"video_rel": "video_file"})
    # Расчётные поля заменяем актуальными значениями без изменения events.csv.
    derived = [c for c in technical.columns if c != "clip_id"]
    analysis = eligible.drop(columns=derived, errors="ignore").merge(technical, on="clip_id", validate="one_to_one")
    analysis = analysis.sort_values("clip_id", key=lambda s: s.map(clip_sort_key)).reset_index(drop=True)
    mapping = duplicate_mapping(inventory)
    mapping.to_csv(processed_dir / "duplicate_map.csv", index=False)
    analysis = analysis.merge(mapping[["clip_id", "canonical_clip_id"]].drop_duplicates("clip_id"),
                              on="clip_id", validate="one_to_one")
    analysis.to_csv(processed_dir / "events_enriched.csv", index=False)
    training_videos = training_video_table(analysis)
    training_videos.to_csv(processed_dir / "training_videos.csv", index=False)
    print(f"Уникальных видео для обучения: {len(training_videos)}; записей событий: {len(analysis)}")
    display(notes)
    print(f"Размечено: {len(events)}; пригодно для анализа: {len(analysis)}; ошибок: {issues['severity'].eq('error').sum()}")
    display(events)
    return events, analysis, training_videos, issues, source_digest


def write_reports(events, readable, frame_report, events_path, source_digest, issues, split, training_videos, split_status, random_state, groups, processed_dir, inventory, generated_figures, data_dir, scene_distance_threshold):
    annotation_ids = set(events["clip_id"].dropna())
    readable_ids = set(readable["clip_id"])
    all_frames_complete = (set(frame_report["clip_id"]) == annotation_ids and not frame_report.empty
                           and frame_report["complete"].all())
    source_unchanged = hashlib.sha256(events_path.read_bytes()).hexdigest() == source_digest
    dataset_valid = not issues["severity"].eq("error").any()
    split_ready = dataset_valid and not split.empty and set(split["clip_id"]) == set(training_videos["clip_id"])
    blockers = []
    if not dataset_valid:
        blockers.append("Video/source annotation errors must be resolved")
    if split.empty:
        blockers.append(split_status)
    p0_complete = dataset_valid and split_ready
    readiness = {"p0_complete": bool(p0_complete), "p0_scope": "agreed preliminary within-room baseline",
                 "split_protocol": "within_room_unique_video", "split_ready": bool(split_ready),
                 "independent_evaluation_ready": False,
                 "limitations": ["Shared room, people and background may appear in both subsets",
                                 "Camera/session identities are unknown; related recordings may cross subsets",
                                 "Metrics do not establish generalization to independent sessions or new cameras",
                                 "Validation set is small and is not a held-out final test set"],
                 "seed": random_state,
                 "split_counts": split.groupby("split").size().to_dict() if not split.empty else {}, "source_records": len(events), "unique_videos": len(training_videos),
                 "positive_events": int(events["source_label"].eq("violation").sum()),
                 "scene_groups": int(groups["group_id"].nunique()), "blockers": blockers,
                 "annotations_sha256": source_digest}
    (processed_dir / "training_readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = {
        "Нет ошибок видео и разметки": not issues["severity"].eq("error").any(),
        "Все размеченные видео найдены и прошли чтение": bool(annotation_ids) and annotation_ids <= readable_ids
            and not inventory["clip_id"].duplicated().any(),
        "Графики параметров и классов построены в этом запуске":
            {"video_parameters.png", "label_distribution.png", "parameters_by_label.png"} <= generated_figures,
        "Все ожидаемые кадры обоих классов прочитаны": bool(all_frames_complete),
        "Разметка и предварительный split готовы (P0 baseline)": p0_complete,
        "Предложение разбиения создано": not split.empty,
        "Исходная разметка не изменилась": source_unchanged,
    }
    status = pd.DataFrame({"Проверка": list(checks), "Готово": list(checks.values())})
    status.to_csv(processed_dir / "eda_checklist.csv", index=False)
    manifest = {"annotations_sha256": source_digest, "video_directory": str(data_dir),
                "video_sha256": dict(zip(inventory["video_rel"], inventory["sha256"])),
                "scene_distance_threshold": scene_distance_threshold, "seed": random_state,
                "split_status": split_status, "checks": {key: bool(value) for key, value in checks.items()}}
    (processed_dir / "eda_run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    display(status)
    assert source_unchanged, "Исходный CSV изменился во время запуска; перезапустите анализ"
    print("Готовность к обучению:", p0_complete)
    print("Блокеры:", blockers)
    print("Метки и время берите из events.csv; training_videos.csv объединяет точные копии. Согласованный предварительный split: split_proposal.csv. Независимость сеансов не подтверждена.")
    return readiness, status
