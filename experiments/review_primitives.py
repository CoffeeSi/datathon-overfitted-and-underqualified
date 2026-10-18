"""Private, timestamped evidence sheets; never generate targets from predictions."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from overfitted_underqualified.ml.dataset import load_records, decode_frames, sha256
from overfitted_underqualified.ml.action_train import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--times", required=True, help="Comma-separated seconds; no target labels")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    events, split = ROOT / "data/annotations/events.csv", ROOT / "data/processed/split_proposal.csv"
    record = next(r for r in load_records(events, split, ROOT / "data/raw/videos") if r.clip_id == args.clip)
    stamps = [float(t) for t in args.times.split(",")]
    if any(not np.isfinite(t) or not 0 <= t <= record.last_time for t in stamps):
        raise ValueError("Timestamp outside video")
    args.output.mkdir(parents=True, exist_ok=True)
    for page, offset in enumerate(range(0, len(stamps), 12), 1):
        times = stamps[offset:offset + 12]
        frames = decode_frames(record, np.array(times))
        # Keep the complete service surface and both people; no upscaling of source detail.
        tiles = []
        for frame, stamp in zip(frames, times):
            im = Image.fromarray(frame.permute(1, 2, 0).numpy())
            im = im.crop((0, int(im.height * .12), im.width, int(im.height * .66)))
            tile = Image.new("RGB", (im.width, im.height + 28), "white")
            tile.paste(im, (0, 28))
            ImageDraw.Draw(tile).text((8, 5), f"{record.clip_id} | {record.split} | {stamp:g}s", fill="black")
            tiles.append(tile)
        w, h = tiles[0].size
        sheet = Image.new("RGB", (w * 3, h * ((len(tiles) + 2) // 3)), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3) * w, (i // 3) * h))
        sheet.save(args.output / f"{record.clip_id}_detail_{page}.jpg", quality=93)
    write_json(args.output / f"{record.clip_id}_evidence.json", dict(
        clip_id=record.clip_id, split=record.split, video_sha256=record.sha256,
        events_sha256=sha256(events), split_sha256=sha256(split), sampled_seconds=stamps,
        created_utc=datetime.now(timezone.utc).isoformat(), method="source-resolution sampled frames; no predictions"))


if __name__ == "__main__":
    main()
