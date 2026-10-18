"""Guard visible-evidence binding independently of binary/model confidence."""
from types import SimpleNamespace
import unittest

from overfitted_underqualified.ml.action_data import ActionSample, mapped_types
from overfitted_underqualified.ml.quality_audit import assess_holdout_candidate, review_basis, summarize_type
from overfitted_underqualified.rules.catalog import event_type_for_observed, supported_event_types
from overfitted_underqualified.rules.evidence import VISUAL_REQUIREMENTS, grounded_rule_match, proposed_mapping


class EvidencePolicyTests(unittest.TestCase):
    def test_every_catalog_type_has_explicit_visual_requirements(self):
        self.assertEqual(set(VISUAL_REQUIREMENTS),supported_event_types())

    def test_missing_cash_history_blocks_type_and_rule_even_with_visible_banknotes(self):
        result=grounded_rule_match('unsecured_accepted_cash',{'banknotes_visible':True},evidence={
            'method':'reviewed_frame_sequence','video_sha256':'a'*64,'basis_id':'synthetic fixture','timestamps_sec':[1,2]})
        self.assertIsNone(result['event_type'])
        self.assertEqual(result['source_links'],[])
        self.assertIn('prior_receipt_observed',result['unknown_visual_facts'])

    def test_binary_confidence_is_never_typed_evidence(self):
        facts={k:True for k in VISUAL_REQUIREMENTS['safe_key_left_in_lock']}
        result=grounded_rule_match('safe_key_left_in_lock',facts,evidence={
            'method':'binary_r3d18','video_sha256':'a'*64,'basis_id':'binary alarm .99','timestamps_sec':[5]})
        self.assertFalse(result['visual_basis_sufficient'])
        self.assertIsNone(grounded_rule_match('violation',{})['event_type'])

    def test_complete_evidence_still_yields_only_conditional_reference(self):
        name='safe_key_left_in_lock'; facts={k:True for k in VISUAL_REQUIREMENTS[name]}
        evidence=dict(method='reviewed_frame_sequence',video_sha256='a'*64,basis_id='SYNTHETIC TEST',timestamps_sec=[1,5])
        result=grounded_rule_match(name,facts,evidence=evidence)
        self.assertEqual(result['event_type'],name)
        self.assertTrue(result['source_links'])
        self.assertTrue(result['requires_review'])
        self.assertIsNone(result['legal_conclusion'])
        self.assertEqual(result['rule_status'],'conditional_reference')
        facts['key_visible_in_lock']=False
        result=grounded_rule_match(name,facts,evidence=evidence)
        self.assertIsNone(result['event_type'])
        self.assertIn('key_visible_in_lock',result['contradicted_visual_facts'])

    def test_action_model_requires_checkpoint_and_quality_approval(self):
        name='safe_key_left_in_lock';facts={k:True for k in VISUAL_REQUIREMENTS[name]}
        evidence=dict(method='evaluated_action_model',video_sha256='a'*64,basis_id='SYNTHETIC TEST',timestamps_sec=[1,5])
        self.assertIsNone(grounded_rule_match(name,facts,evidence=evidence)['event_type'])
        evidence.update(checkpoint_sha256='b'*64,quality_approved=True)
        self.assertEqual(grounded_rule_match(name,facts,evidence=evidence)['event_type'],name)

    def test_semantic_gap_is_blocked_in_catalog_and_training_proposals(self):
        for tag in ('unattended_bag_contents_unknown','employee_steps_away_suspected_cash_on_desk'):
            self.assertEqual(proposed_mapping(tag)['status'],'needs_context_review')
            self.assertIsNone(event_type_for_observed(tag))
            self.assertEqual(mapped_types(tag),[])

    def test_alias_points_do_not_become_independent_positive_videos(self):
        name='unsecured_accepted_cash'
        record=SimpleNamespace(clip_id='X-4',split='train')
        review=dict(present_types=[name],evidence_note='Batch-supplied labels',reviewer='dataset_owner')
        samples=[ActionSample(record,center,(name,),False,(alias,),reviews=(review,))
                 for center,alias in ((57,'X-4'),(68,'X-47'))]
        result=summarize_type(name,samples)
        self.assertEqual(result['train']['positive_windows'],2)
        self.assertEqual(result['train']['positive_ids'],['X-4'])
        self.assertIn('single_canonical_train_positive',result['quality_limitations'])
        self.assertEqual(result['train']['non_batch_positive_ids'],[])
        self.assertFalse(result['deployment_approved'])

    def test_batch_review_is_not_reported_as_frame_review(self):
        self.assertEqual(review_basis(dict(evidence_note='Batch-supplied window labels; no independent frame-by-frame review.',
            reviewer='dataset_owner')),'batch_supplied_labels')

    def test_holdout_requires_real_unseen_group_and_reviewed_labels(self):
        candidate=dict(sha256='a'*64,origin='real_recording',source_group='new-session',
            source_group_verified=True,previously_used=False,labels_reviewed=True,derivative_of=None)
        result=assess_holdout_candidate(candidate,used_hashes={'b'*64},used_source_groups={'known-session'})
        self.assertTrue(result['eligible'])
        self.assertIsNone(result['assigned_partition'])
        for change in (dict(origin='synthetic_generated'),dict(source_group='known-session'),
                       dict(sha256='b'*64),dict(derivative_of='X-4'),dict(source_group_verified=False),
                       dict(previously_used=None),dict(labels_reviewed=False)):
            with self.subTest(change=change):
                result=assess_holdout_candidate({**candidate,**change},used_hashes={'b'*64},used_source_groups={'known-session'})
                self.assertFalse(result['eligible'])


if __name__=='__main__':unittest.main()
