"""Research-only KVMS evidence intake tests. No audio hardware dependencies."""
import copy
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from perceptual.kvms_evidence import InvalidEvidence, summarize_evidence


RECORD = {
    'schema_version':'kvms-blind-result/1.1',
    'experiment_id':'PENA-KVMS-GTR-DOSE-03',
    'track':'Логопед — Пена',
    'blinding':'key_revealed_after_listener_choices',
    'listener_n':1,
    'blind_key':{
        'S1':{'A':'v04','B':'v06','C':'v05'},
        'S2':{'A':'v06','B':'v05','C':'v04'},
        'S3':{'A':'v05','B':'v04','C':'v06'},
    },
    'revealed_choices':[
        {'id':'S1','start_s':26,'end_s':46,'selected_letter':'C','selected_mix':'v05'},
        {'id':'S2','start_s':52,'end_s':72,'selected_letter':'B','selected_mix':'v05'},
        {'id':'S3','start_s':172,'end_s':192,'selected_letter':'A','selected_mix':'v05'},
    ],
    'levels':{'actual_max_within_section_lufs_difference_below':0.001},
    'signal_integrity':{'all_nine_mp3_hashes_match_existing_verification':True},
}


class KVMSIntakeTests(unittest.TestCase):
    def test_prefers_v05_at_all_three_sections(self):
        r=summarize_evidence(RECORD)
        self.assertEqual(r.unanimous_choice,'v05')
        self.assertEqual(r.counts_by_mix,{'v05':3})
        self.assertEqual(r.sections,3)
        self.assertEqual(r.independent_listeners,1)
        self.assertEqual(r.study_class,'exploratory_consistent')
        self.assertFalse(r.can_auto_apply)

    def test_swapped_blind_key_rejected(self):
        d=copy.deepcopy(RECORD)
        d['blind_key']['S1']['C']='v04'
        with self.assertRaises(InvalidEvidence):summarize_evidence(d)

    def test_duplicate_section_rejected(self):
        d=copy.deepcopy(RECORD)
        d['revealed_choices'][1]['id']='S1'
        with self.assertRaises(InvalidEvidence):summarize_evidence(d)

    def test_missing_loudness_is_rejected(self):
        d=copy.deepcopy(RECORD)
        del d['levels']
        with self.assertRaises(InvalidEvidence):summarize_evidence(d)

    def test_loudness_mismatch_withholds_candidate(self):
        d=copy.deepcopy(RECORD)
        d['levels']['actual_max_within_section_lufs_difference_below']=1.2
        r=summarize_evidence(d)
        self.assertEqual(r.study_class,'exploratory_inconclusive')
        self.assertIsNone(r.candidate_for_future_validation)
        self.assertFalse(r.can_auto_apply)

    def test_integrity_mismatch_withholds_candidate(self):
        d=copy.deepcopy(RECORD)
        d['signal_integrity']['all_nine_mp3_hashes_match_existing_verification']=False
        r=summarize_evidence(d)
        self.assertIsNone(r.candidate_for_future_validation)
        self.assertFalse(r.can_auto_apply)

    def test_new_generic_integrity_field(self):
        d=copy.deepcopy(RECORD)
        d['signal_integrity']={'audio_hashes_verified':True}
        r=summarize_evidence(d)
        self.assertTrue(r.integrity_verified)

    def test_incomplete_blinding_rejected(self):
        d=copy.deepcopy(RECORD)
        d['blinding']='key_revealed_before_vote'
        with self.assertRaises(InvalidEvidence):summarize_evidence(d)

    def test_nonunanimous_is_not_calibrated(self):
        d=copy.deepcopy(RECORD)
        d['revealed_choices'][0]['selected_letter']='A'
        d['revealed_choices'][0]['selected_mix']='v04'
        r=summarize_evidence(d)
        self.assertIsNone(r.unanimous_choice)
        self.assertFalse(r.can_auto_apply)

    def test_nan_level_bounds_rejected(self):
        d=copy.deepcopy(RECORD)
        d['levels']['actual_max_within_section_lufs_difference_below']=float('nan')
        with self.assertRaises(InvalidEvidence):summarize_evidence(d)

    def test_boolean_listener_n_rejected(self):
        d=copy.deepcopy(RECORD)
        d['listener_n']=True
        with self.assertRaises(InvalidEvidence):summarize_evidence(d)


if __name__=='__main__':unittest.main()
