"""Supplemental import must never change partitions or manufacture reviewed labels."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from overfitted_underqualified.ml.supplemental_data import (
    audit_supplemental_data, read_workbook, resolve_augmentations, sha256,
)


META = dict(fps=1., frame_count=100, width=640, height=480, duration_sec=100.)
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_workbook(path, inventory, claims, formula=False):
    with ZipFile(path, "w") as archive:
        for number, rows in enumerate((inventory, claims), 1):
            root = ET.Element("worksheet", xmlns=NS)
            sheet = ET.SubElement(root, "sheetData")
            headers = list(rows[0])
            values = [headers] + [[row[h] for h in headers] for row in rows]
            for row_index, values_row in enumerate(values, 6):
                row = ET.SubElement(sheet, "row", r=str(row_index))
                for index, value in enumerate(values_row):
                    cell = ET.SubElement(row, "c", r=f"{chr(65 + index)}{row_index}", t="inlineStr")
                    ET.SubElement(ET.SubElement(cell, "is"), "t").text = str(value)
                    if formula and row_index == 7 and index == 0:
                        ET.SubElement(cell, "f").text = 'HYPERLINK("bad", "bad")'
            archive.writestr(f"xl/worksheets/sheet{number}.xml", ET.tostring(root))


class SupplementalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original, self.augmented, self.synthetic = [self.root / n for n in ("raw", "aug", "synthetic")]
        for path in (self.original, self.augmented, self.synthetic):
            path.mkdir()
        for clip, content in (("X-4", b"canonical-four"), ("X-47", b"canonical-four"), ("X-2", b"validation-two")):
            (self.original / f"{clip}.mov").write_bytes(content)
        self.events = self.root / "events.csv"
        self.split = self.root / "split.csv"
        self.labels = self.root / "labels.csv"
        self.workbook = self.root / "annotations.xlsx"
        write_csv(self.events, [dict(clip_id=n, source_label="violation", timecode=t)
                                for n, t in (("X-4", "00:57"), ("X-47", "01:08"), ("X-2", "00:10"))])
        write_csv(self.split, [dict(clip_id=n, source_label="violation", split=subset,
                                   sha256=sha256(self.original / f"{n}.mov"),
                                   source_clip_ids=json.dumps(ids), event_times_sec=json.dumps(times))
                               for n, subset, ids, times in (("X-4", "train", ["X-4", "X-47"], [57., 68.]),
                                                            ("X-2", "validation", ["X-2"], [10.]))])
        self.label_rows = []
        for source, stamp in (("X-4", 57), ("X-47", 68), ("X-2", 10)):
            for augmentation in ("mirror", "warm_sepia"):
                name = f"{source}__{augmentation}.mov"
                (self.augmented / name).write_bytes(name.encode())
                self.label_rows.append({"наименование файла": name, "таймстемп": f"{stamp} секунда",
                                        "нарушение / без нарушения": "нарушение", "source_file": f"{source}.mov",
                                        "source_group": source, "augmentation": augmentation})
        write_csv(self.labels, self.label_rows)
        video = self.synthetic / "S01_original.mp4"
        video.write_bytes(b"generated-video")
        (self.synthetic / "scenario_01_clean.mov").write_bytes(b"unlabelled")
        self.inventory = [dict(video_id="S01_original", source_id="S01", augmentation="none",
                               **{"Файл": "videos/originals/S01_original.mp4", "FPS": "1", "Кадров": "100",
                                  "Ширина, px": "640", "Высота, px": "480", "Длительность, с": "100",
                                  "Событий": "1", "SHA-256": sha256(video), "Примечание": "Generated scenario"})]
        self.claims = [dict(video_id="S01_original", source_id="S01", augmentation="none",
                            **{"FPS": "1", "Начало, с": "1", "Конец, с": "2", "start_frame": "1",
                               "end_frame_exclusive": "2", "Пункт закона": "unverified source claim"})]
        write_workbook(self.workbook, self.inventory, self.claims)
        self.metadata = patch("overfitted_underqualified.ml.supplemental_data.inspect_metadata", return_value=dict(META))
        self.metadata.start(); self.addCleanup(self.metadata.stop)

    def audit(self):
        return audit_supplemental_data(events_path=self.events, split_path=self.split, video_dir=self.original,
                                      augmented_dir=self.augmented, labels_path=self.labels,
                                      synthetic_dir=self.synthetic, workbook_path=self.workbook)

    def sample(self, clip="X-4", subset="train"):
        record = SimpleNamespace(clip_id=clip, split=subset, last_time=99.,
                                 sha256=sha256(self.original / f"{clip}.mov"))
        return SimpleNamespace(record=record, center_sec=57. if clip == "X-4" else 10.,
                               positives=("banknotes_on_counter",), negatives=())

    def resolve(self, manifest, sample=None, **kwargs):
        return resolve_augmentations(manifest, sample or self.sample(), events_path=self.events,
                                     split_path=self.split, **kwargs)

    def test_fixed_split_alias_inheritance_and_synthetic_isolation(self):
        before = {p: p.read_bytes() for p in (self.events, self.split, self.labels, self.workbook)}
        manifest = self.audit()
        self.assertEqual(manifest["counts"]["augmentations"],
                         {"eligible_train_parent": 4, "quarantined_validation_descendant": 2})
        aliases = [a for a in manifest["augmentations"] if a.get("source_id") == "X-47"]
        self.assertTrue(all(a["parent_id"] == "X-4" and a["inherited_time_sec"] == 68 for a in aliases))
        synthetic = manifest["synthetic"][0]
        self.assertEqual(synthetic["role"], "auxiliary_train_only")
        self.assertFalse(synthetic["training_eligible"])
        self.assertFalse(synthetic["label_import_authorized"])
        self.assertFalse(synthetic["independent_evidence"])
        self.assertEqual(manifest["synthetic"][1]["status"], "quarantined_unlabelled")
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_consistency_and_unsafe_path_fail_closed_per_row(self):
        mutations = [("таймстемп", "00:56"), ("нарушение / без нарушения", "без нарушения"),
                     ("source_group", "X-999"), ("source_file", "../X-4.mov"),
                     ("наименование файла", "../X-4__mirror.mov"),
                     ("наименование файла", "missing.mov")]
        original = dict(self.label_rows[0])
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                self.label_rows[0] = dict(original, **{key: value})
                write_csv(self.labels, self.label_rows)
                first = self.audit()["augmentations"][0]
                self.assertEqual(first["status"], "rejected")
                self.assertFalse(first["training_eligible"])
        self.label_rows[0] = original

    def test_metadata_drift_and_duplicate_filename_rejected(self):
        self.label_rows.append(dict(self.label_rows[0]))
        write_csv(self.labels, self.label_rows)
        manifest = self.audit()
        self.assertEqual(manifest["augmentations"][-1]["status"], "rejected")
        self.assertEqual(manifest["augmentations"][0]["status"], "rejected")
        with patch("overfitted_underqualified.ml.supplemental_data.inspect_metadata",
                   side_effect=lambda p: dict(META, frame_count=99) if Path(p).parent == self.augmented else dict(META)):
            manifest = self.audit()
        self.assertTrue(all(a["status"] == "rejected" for a in manifest["augmentations"]))

    def test_exact_duplicate_across_partitions_rejects_both_copies(self):
        train = self.augmented / "X-4__mirror.mov"
        validation = self.augmented / "X-2__mirror.mov"
        validation.write_bytes(train.read_bytes())
        manifest = self.audit()
        rows = [r for r in manifest["augmentations"] if r.get("path") in {str(train), str(validation)}]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["status"] == "rejected" and not r["training_eligible"] for r in rows))

    def test_resolver_deduplicates_aliases_and_inverts_mirror_before_crop(self):
        manifest = self.audit()
        self.assertEqual([a.augmentation for a in self.resolve(manifest)], ["warm_sepia"])
        items = self.resolve(manifest, inverse_mirror=True)
        self.assertEqual({a.augmentation for a in items}, {"mirror", "warm_sepia"})
        self.assertTrue(all(a.path.name.startswith("X-4__") for a in items))
        class Frames:
            def flip(self, dimension):
                return ("flipped", dimension)
        mirror = next(a for a in items if a.inverse_horizontal_flip)
        self.assertEqual(mirror.restore_orientation(Frames()), ("flipped", -1))
        with self.assertRaisesRegex(ValueError, "CHW"):
            mirror.restore_orientation(SimpleNamespace(ndim=3, shape=(20, 30, 3)))
        ordinary = next(a for a in items if not a.inverse_horizontal_flip)
        data = Frames()
        self.assertIs(ordinary.restore_orientation(data), data)

    def test_resolver_rejects_validation_unknown_targets_and_changed_content(self):
        manifest = self.audit()
        with self.assertRaisesRegex(ValueError, "Validation"):
            self.resolve(manifest, self.sample("X-2", "validation"))
        sample = self.sample(); sample.positives = ()
        with self.assertRaisesRegex(ValueError, "reviewed"):
            self.resolve(manifest, sample)
        (self.augmented / "X-4__warm_sepia.mov").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.resolve(manifest)

    def test_resolver_rejects_stale_source_or_split(self):
        for source in (self.events, self.split):
            with self.subTest(source=source.name):
                manifest = self.audit(); original = source.read_bytes()
                source.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "current data/split"):
                    self.resolve(manifest)
                source.write_bytes(original)

    def test_workbook_claims_are_data_and_hash_timing_and_paths_are_checked(self):
        inventory, claims = read_workbook(self.workbook)
        self.assertEqual(claims[0]["Пункт закона"], "unverified source claim")
        for field, value in (("SHA-256", "0" * 64), ("Файл", "../S01_original.mp4"), ("Кадров", "99")):
            with self.subTest(field=field):
                row = dict(self.inventory[0], **{field: value})
                write_workbook(self.workbook, [row], self.claims)
                self.assertEqual(self.audit()["synthetic"][0]["status"], "rejected")
        claim = dict(self.claims[0], **{"start_frame": "2"})
        write_workbook(self.workbook, self.inventory, [claim])
        self.assertEqual(self.audit()["synthetic"][0]["status"], "rejected")
        write_workbook(self.workbook, self.inventory, self.claims, formula=True)
        with self.assertRaisesRegex(ValueError, "Formula"):
            read_workbook(self.workbook)

    def test_corrupt_original_split_cannot_be_bypassed_by_supplemental_labels(self):
        (self.original / "X-47.mov").write_bytes(b"different alias")
        with self.assertRaisesRegex(ValueError, "Duplicate alias"):
            self.audit()

    def test_cyrillic_source_ids_and_worded_minutes_preserve_alias_times(self):
        (self.original / "X-47.mov").rename(self.original / "Х-47.mov")
        for row in self.label_rows:
            if row["source_group"] == "X-47":
                path = self.augmented / row["наименование файла"]
                name = path.name.replace("X", "Х")
                path.rename(self.augmented / name)
                row.update({"наименование файла": name, "source_group": "Х-47", "source_file": "Х-47.mov",
                            "таймстемп": "1минута 8 секунда"})
        write_csv(self.labels, self.label_rows)
        manifest = self.audit()
        aliases = [r for r in manifest["augmentations"] if r.get("source_id") == "X-47"]
        self.assertEqual(len(aliases), 2)
        self.assertTrue(all(r["status"] == "eligible_train_parent" and r["inherited_time_sec"] == 68 for r in aliases))


if __name__ == "__main__":
    unittest.main()
