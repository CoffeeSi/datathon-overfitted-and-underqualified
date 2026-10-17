"""EDA figures and contact sheets; output paths are explicit arguments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from IPython.display import display
from tqdm.auto import tqdm
from .video import read_frames
from .annotations import LABELS
# Сохранение и показ графиков; используется также в следующих разделах.

def save_figure(fig, filename, figures_dir):
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Моменты кадров вокруг события с учётом границ видео.

def window_timestamps(center, duration, fps, window=4.0, step=0.5):
    if not all(np.isfinite(v) for v in (center, duration, fps, window, step)):
        raise ValueError("Window parameters must be finite")
    if duration <= 0 or fps <= 0 or step <= 0 or window < 0 or not 0 <= center < duration:
        raise ValueError("Invalid event time or video/window parameters")
    last = max(0.0, duration - 1 / fps)
    center = min(float(center), last)
    offsets = np.arange(-window, window + 1e-8, step)
    stamps = np.clip(np.r_[center + offsets, center], 0, last)
    return sorted({round(float(t), 6) for t in stamps})

# Контактный лист для ручного просмотра и отчёт о прочитанных кадрах.

def save_contact_sheet(clip_id, path, stamps, folder, show=False, root=None):
    frames = read_frames(path, stamps)
    columns = min(5, len(stamps))
    rows = int(np.ceil(len(stamps) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 2.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, stamp, frame in zip(axes.flat, stamps, frames):
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, "Кадр не прочитан", ha="center", transform=ax.transAxes)
        ax.set_title(f"{stamp:.2f} s")
    fig.suptitle(clip_id)
    target = folder / f"{clip_id}.png"
    fig.tight_layout()
    fig.savefig(target, dpi=90, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return {"clip_id": clip_id, "expected": len(stamps), "decoded": sum(f is not None for f in frames),
            "complete": all(f is not None for f in frames), "output": str(target.relative_to(root) if root is not None else target)}


def plot_distributions(readable, events, analysis, figures_dir):
    generated_figures = set()
    if not readable.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(readable["duration_sec"], bins=12)
        axes[0].set(xlabel="Длительность, сек", ylabel="Видео")
        readable["fps"].round(2).value_counts().sort_index().plot.bar(ax=axes[1], title="FPS")
        resolution = readable["width"].astype(str) + "×" + readable["height"].astype(str)
        resolution.value_counts().plot.bar(ax=axes[2], title="Разрешение")
        save_figure(fig, "video_parameters.png", figures_dir)
        generated_figures.add("video_parameters.png")
        display(readable[["duration_sec", "fps", "width", "height"]].describe())

    label_counts = events["source_label"].value_counts().reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    label_counts.plot.bar(ax=ax, ylabel="Число роликов", title="Исходные метки")
    save_figure(fig, "label_distribution.png", figures_dir)
    generated_figures.add("label_distribution.png")
    print(label_counts.to_string())
    if label_counts.min() > 0:
        print(f"Отношение большего класса к меньшему: {label_counts.max() / label_counts.min():.2f}")
    else:
        print("Один из классов отсутствует; обучение и валидация бинарной модели пока невозможны.")

    categories = events.loc[events["source_label"].eq("violation"), "candidate_violation_type"].dropna().value_counts()
    if not categories.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        categories.sort_values().plot.barh(ax=ax, xlabel="Число роликов", title="Предварительные категории (не подтверждены)")
        save_figure(fig, "candidate_violation_types.png", figures_dir)
        generated_figures.add("candidate_violation_types.png")
    types = events.loc[events["source_label"].eq("violation"), "observed_event_type"].value_counts()
    print(f"Предварительных типов событий: {len(types)}; с единственным примером: {int(types.eq(1).sum())}")
    timed = analysis.loc[analysis["source_label"].eq("violation") & analysis["timecode_sec"].notna()]
    if not timed.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(timed["timecode_sec"], bins=10)
        axes[0].set(xlabel="Таймкод, сек", ylabel="События")
        axes[1].hist(timed["timecode_sec"] / timed["duration_sec"], bins=10, range=(0, 1))
        axes[1].set(xlabel="Положение в ролике (0 — начало, 1 — конец)")
        save_figure(fig, "timecode_distribution.png", figures_dir)
        generated_figures.add("timecode_distribution.png")
    return generated_figures


def plot_class_parameters(analysis, processed_dir, figures_dir, generated_figures):
    unique_videos = analysis.drop_duplicates("canonical_clip_id")
    if not unique_videos.empty:
        summary = unique_videos.groupby("source_label")[["duration_sec", "fps", "brightness", "blur_score"]].agg(["count", "median", "min", "max"])
        summary.to_csv(processed_dir / "class_parameter_summary.csv")
        display(summary)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, column in zip(axes, ["duration_sec", "brightness", "blur_score"]):
            sns.boxplot(data=unique_videos, x="source_label", y=column, ax=ax)
            sns.stripplot(data=unique_videos, x="source_label", y=column, ax=ax, color="black", size=3)
        save_figure(fig, "parameters_by_label.png", figures_dir)
        generated_figures.add("parameters_by_label.png")
        for column in ("fps", "width", "height"):
            print(column)
            display(pd.crosstab(unique_videos[column], unique_videos["source_label"]))


def contact_sheets(analysis, data_dir, event_frames_dir, normal_frames_dir, processed_dir, window_sec=4.0, frame_step_sec=0.5):
    frame_records, shown = [], {label: 0 for label in LABELS}
    for row in tqdm(list(analysis.itertuples(index=False)), desc="contact sheets"):
        if row.source_label == "violation":
            stamps = window_timestamps(row.timecode_sec, row.duration_sec, row.fps, window_sec, frame_step_sec)
            folder = event_frames_dir
        else:
            stamps = np.linspace(0, max(0, min(row.duration_sec - 1 / row.fps, row.duration_sec * 0.98)), 5).tolist()
            folder = normal_frames_dir
        record = save_contact_sheet(row.clip_id, data_dir / row.video_file, stamps, folder, shown[row.source_label] < 2)
        record["source_label"] = row.source_label
        frame_records.append(record)
        shown[row.source_label] += 1
    frame_report = pd.DataFrame(frame_records, columns=["clip_id", "expected", "decoded", "complete", "output", "source_label"])
    frame_report.to_csv(processed_dir / "frame_extraction.csv", index=False)
    return frame_report


def plot_zones(zones_path, analysis, data_dir, figures_dir):
    zones = json.loads(zones_path.read_text(encoding="utf-8")) if zones_path.exists() else {}
    for clip_id, clip_zones in zones.items():
        match = analysis.loc[analysis["clip_id"].eq(clip_id)]
        if match.empty:
            print(f"{clip_id}: нет пригодного видео")
            continue
        row = match.iloc[0]
        stamp = row["timecode_sec"] if pd.notna(row["timecode_sec"]) else row["duration_sec"] / 2
        frame = read_frames(data_dir / row["video_file"], [min(stamp, row["duration_sec"] - 1 / row["fps"])])[0]
        if frame is None:
            print(f"{clip_id}: кадр не прочитан")
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(frame)
        for name, box in clip_zones.items():
            if len(box) != 4 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box):
                raise ValueError(f"Некорректная зона {clip_id}/{name}")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                raise ValueError(f"Зона вне кадра: {clip_id}/{name}")
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red"))
            ax.text(x1, y1, name, color="red")
        ax.axis("off")
        save_figure(fig, f"zones_{clip_id}.png", figures_dir)
    if not zones:
        print("Зоны пока не размечены — дополнительный этап пропущен.")


def plot_scene_candidates(pairs, thumbnails, figures_dir):
    if not pairs.empty:
        preview = pairs.head(6)
        fig, axes = plt.subplots(len(preview), 2, figsize=(10, 3 * len(preview)), squeeze=False)
        for row_axes, pair in zip(axes, preview.itertuples(index=False)):
            for ax, clip_id in zip(row_axes, (pair.clip_a, pair.clip_b)):
                ax.imshow(np.median(np.stack(thumbnails[clip_id]), axis=0), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{clip_id}; distance={pair.scene_distance:.3f}")
                ax.axis("off")
        save_figure(fig, "scene_candidates_preview.png", figures_dir)
