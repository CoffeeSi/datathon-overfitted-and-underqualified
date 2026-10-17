"""Notebook views of raw sampled frames and recorded experiment results."""
import matplotlib.pyplot as plt
import numpy as np

from .dataset import decode_frames, sample_times


def show_windows(dataset, indices, titles=None):
    """Show selected frames before normalization; use exactly the dataset's sampler."""
    indices = list(indices)
    if not indices:
        print("Нет окон для просмотра.")
        return
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        record, center = dataset.samples[index]
        times = sample_times(center, record.last_time, dataset.window_sec, dataset.frames)
        frames = decode_frames(record, times)
        for ax, frame_index in zip(axes[row], (0, 5, 10, 15)):
            ax.imshow(frames[frame_index].permute(1, 2, 0).numpy())
            ax.axis("off")
            prefix = titles[row] if titles is not None else f"{record.clip_id}; label={record.label}"
            ax.set_title(f"{prefix}\n{times[frame_index]:.2f}s")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_history(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    for subset in ("train", "validation"):
        ax.plot([r["epoch"] for r in history], [r[subset]["loss"] for r in history], marker="o", label=subset)
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_confusion(metrics):
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    names = ["no_violation", "violation"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=names, yticklabels=names,
           xlabel="Prediction (threshold 0.5)", ylabel="Source label")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="red")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
