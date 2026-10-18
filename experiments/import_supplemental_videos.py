"""Copy external sources into ignored data/raw without changing originals or labels."""
import argparse
import json
from pathlib import Path
import shutil

from overfitted_underqualified.ml.supplemental_data import audit_supplemental_data, sha256, VIDEO_SUFFIXES
ROOT = Path(__file__).resolve().parents[1]


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f"Different existing source: {destination}")
    else:
        shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise ValueError(f"Copy hash mismatch: {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('augmented-dir', 'synthetic-dir', 'labels', 'workbook'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'data/raw/supplemental')
    args = parser.parse_args()
    for directory, target in ((args.augmented_dir, 'augmentations'), (args.synthetic_dir, 'synthetic')):
        for source in sorted(directory.iterdir()):
            if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
                copy_verified(source, args.output_root/target/source.name)
    for source, name in ((args.labels,'labels_augmented.csv'),(args.workbook,'annotations.xlsx')):
        copy_verified(source,args.output_root/'source_metadata'/name)
    audit = audit_supplemental_data(
        events_path=ROOT/'data/annotations/events.csv',split_path=ROOT/'data/processed/split_proposal.csv',
        video_dir=ROOT/'data/raw/videos',augmented_dir=args.output_root/'augmentations',
        labels_path=args.output_root/'source_metadata/labels_augmented.csv',
        synthetic_dir=args.output_root/'synthetic',workbook_path=args.output_root/'source_metadata/annotations.xlsx')
    path=args.output_root/'audit.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        for value in (previous,audit):value.pop('created_utc',None)
        if previous != audit:raise FileExistsError('Existing audit differs; use a new output-root')
    else:path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=str(path),counts=audit['counts']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
