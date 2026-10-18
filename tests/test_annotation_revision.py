"""A registered legal-only edit preserves binary evaluation without permitting drift."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from overfitted_underqualified.ml.dataset import sha256,verify_annotation_revision,annotation_fields_digest
from overfitted_underqualified.rules.catalog import load_catalog,source_links_for,supported_event_types


class AnnotationRevisionTests(unittest.TestCase):
    def test_action_review_revision_preserves_binary_supervision(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.csv'; registry=Path(d)/'revisions.json'
            rows=[dict(clip_id='X-1',source_label='violation',timecode='00:05',action_review='{}')]
            fields=list(rows[0]); protected=fields[:-1]
            def write():
                with path.open('w',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
            write();old=sha256(path);rows[0]['action_review']='{"present_types":["safe_key_left_in_lock"]}';write()
            entry=dict(schema='binary_action_review_revision_v1',before_sha256=old,after_sha256=sha256(path),
                       allowed_changed_fields=['action_review'],preserved_fields=protected,
                       preserved_fields_sha256=annotation_fields_digest(rows,protected))
            registry.write_text(json.dumps([entry]),encoding='utf-8')
            self.assertEqual(verify_annotation_revision(path,old,registry_path=registry),entry)
            rows[0]['timecode']='00:06';write();entry['after_sha256']=sha256(path)
            registry.write_text(json.dumps([entry]),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'protected'):verify_annotation_revision(path,old,registry_path=registry)
            entry['allowed_changed_fields']=['action_review','timecode']
            registry.write_text(json.dumps([entry]),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'scope'):verify_annotation_revision(path,old,registry_path=registry)

    def test_description_revision_does_not_allow_supervision_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.csv';registry=Path(d)/'revisions.json'
            rows=[dict(clip_id='X-1',observed_event='not described',source_label='no_violation',timecode='',legal_basis='')]
            fields=list(rows[0]);protected=[k for k in fields if k!='observed_event']
            def write():
                with path.open('w',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
            write();old=sha256(path);rows[0]['observed_event']='Cashier uses a computer';write()
            entry=dict(schema='binary_metadata_revision_v1',before_sha256=old,after_sha256=sha256(path),
                       allowed_changed_fields=['observed_event'],preserved_fields=protected,
                       preserved_fields_sha256=annotation_fields_digest(rows,protected))
            registry.write_text(json.dumps([entry]),encoding='utf-8')
            self.assertEqual(verify_annotation_revision(path,old,registry_path=registry),entry)
            rows[0]['source_label']='violation';write();entry['after_sha256']=sha256(path)
            registry.write_text(json.dumps([entry]),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'protected'):
                verify_annotation_revision(path,old,registry_path=registry)
            entry['allowed_changed_fields']=['source_label'];registry.write_text(json.dumps([entry]),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'scope'):
                verify_annotation_revision(path,old,registry_path=registry)

    def test_registered_pair_protects_every_nonlegal_value(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.csv';registry=Path(d)/'revisions.json'
            rows=[dict(clip_id='X-1',source_label='violation',timecode='00:01',legal_basis='old')]
            fields=list(rows[0]);protected=fields[:-1]
            def write():
                with path.open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
            write();old=sha256(path)
            self.assertIsNone(verify_annotation_revision(path,old,registry_path=registry))
            rows[0]['legal_basis']='№231, п. 20';write()
            with self.assertRaises(ValueError):verify_annotation_revision(path,old,registry_path=registry)
            entry=dict(schema='legal_basis_only_revision_v1',before_sha256=old,after_sha256=sha256(path),
                       allowed_changed_fields=['legal_basis'],preserved_fields=protected,
                       preserved_fields_sha256=annotation_fields_digest(rows,protected))
            registry.write_text(json.dumps([entry]),encoding='utf-8')
            self.assertEqual(verify_annotation_revision(path,old,registry_path=registry),entry)
            rows[0]['timecode']='00:02';write()
            with self.assertRaises(ValueError):verify_annotation_revision(path,old,registry_path=registry)
            # Even an accidentally updated file hash cannot conceal protected changes.
            entry['after_sha256']=sha256(path);registry.write_text(json.dumps([entry]),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'protected'):verify_annotation_revision(path,old,registry_path=registry)

    def test_research_sources_do_not_implicitly_enable_action_types(self):
        catalog=load_catalog()
        self.assertEqual({s['id'] for s in catalog['sources']},{'nb_231','nb_14','arrfm_53'})
        self.assertFalse(any(p['source_id']=='arrfm_53' for p in catalog['provisions']))
        self.assertEqual(next(p for p in catalog['provisions'] if p['id']=='nb_14_p8')['usage'],'research_context_only')
        for name in supported_event_types():
            self.assertTrue(source_links_for(name))
            self.assertTrue(all('/V1900019680' in link['url'] for link in source_links_for(name)))
        with (ROOT/'data/annotations/events.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        x20=next(r for r in rows if r['clip_id']=='X-20')
        self.assertEqual((x20['source_label'],float(x20['timecode_sec'])),('violation',5.0))
        self.assertIn('№53',next(r for r in rows if r['clip_id']=='X-28')['legal_basis'])
        revisions=json.loads((ROOT/'data/annotation_revisions.json').read_text(encoding='utf-8'))
        current=sha256(ROOT/'data/annotations/events.csv')
        expected=next(r for r in revisions if r['before_sha256']==revisions[0]['before_sha256'] and r['after_sha256']==current)
        self.assertEqual(verify_annotation_revision(ROOT/'data/annotations/events.csv',revisions[0]['before_sha256']),expected)


if __name__=='__main__':unittest.main()
