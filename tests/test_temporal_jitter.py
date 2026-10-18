"""Point labels remain inside jitter windows; no validation/negative relabelling."""
from types import SimpleNamespace
import unittest

from overfitted_underqualified.ml.temporal_jitter import positive_jitter,balanced_probabilities


class TemporalJitterTests(unittest.TestCase):
    def record(self,clip='X-4',label=1,split='train'):
        return SimpleNamespace(clip_id=clip,label=label,split=split,last_time=100.,aliases=('X-4','X-47'))

    def test_distinct_alias_events_remain_anchors_and_negatives_not_added(self):
        record=self.record();negative=self.record('X-3',0)
        base=SimpleNamespace(samples=[(record,57.),(record,68.),(negative,50.)],window_sec=4.,frames=16)
        extra,lineage=positive_jitter(base,[-1.,1.])
        self.assertEqual([t for _,t in extra.samples],[56.,58.,67.,69.])
        self.assertEqual([p['event_anchor_sec'] for p in lineage],[57.,57.,68.,68.])
        self.assertEqual(len(base.samples),3)
        self.assertTrue(all(p['support_start_sec']<=p['event_anchor_sec']<=p['support_end_sec'] for p in lineage))

    def test_validation_and_excessive_shifts_fail(self):
        base=SimpleNamespace(samples=[(self.record(split='validation'),10.)],window_sec=4.,frames=16)
        with self.assertRaises(ValueError):positive_jitter(base,[-1.,1.])
        base.samples=[(self.record(),10.)]
        with self.assertRaises(ValueError):positive_jitter(base,[3.])

    def test_edge_clipping_does_not_duplicate_original_or_lose_anchor(self):
        record=self.record();base=SimpleNamespace(samples=[(record,0.)],window_sec=4.,frames=16)
        extra,lineage=positive_jitter(base,[-1.,1.])
        self.assertEqual([t for _,t in extra.samples],[1.])
        self.assertEqual(lineage[0]['event_anchor_sec'],0.)

    def test_more_positive_windows_do_not_change_class_mass(self):
        variants=['original','mirror','warm_sepia','cool_camera','cctv_low_quality']
        sources=[dict(clip_id=c,label=y,center_sec=t,variant=v,split='train')
                 for c,y,times in [('X-4',1,[56.,57.,58.,67.,68.,69.]),('X-3',0,[2.,50.,98.])]
                 for t in times for v in variants]
        p=balanced_probabilities(sources,.8)
        self.assertAlmostEqual(sum(float(w) for s,w in zip(sources,p) if s['label']==1),.5,places=6)
        self.assertAlmostEqual(sum(float(w) for s,w in zip(sources,p) if s['variant']=='original'),.8,places=6)


if __name__=='__main__':unittest.main()
