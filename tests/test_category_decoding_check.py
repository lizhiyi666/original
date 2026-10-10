import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np

from experiment_io import sha256_file
from tools.run_category_decoding_check import subset_inputs,compare_pair,verify_delivery


class CategoryDiagnosticTests(unittest.TestCase):
    def inputs(self):
        items=[dict(indices=list(range(i,i+64)),global_start=i,batch=types.SimpleNamespace(batch_size=64))
               for i in (0,64,128)]
        traces=[dict(indices=v['indices'],global_start=v['global_start'],spatial_rng_before='a',spatial_rng_after='b',
                     projection_rng_after='p',distance_rng_before='d0',distance_rng_after='d1') for v in items]
        records=[dict(arrival_times=np.array([.2,.4]),marks=np.array([4,5]),checkins=np.array([7,9]),
                      gps=[[0.,0.],[1.,1.]],condition1_indicator=np.array([1,0])) for _ in range(192)]
        return dict(indices=list(range(192)),batches=items,manifest_sha256='old',result={'samples':192}),dict(sequences=records,batch_traces=traces)

    def test_exact_original_batches_are_reused_without_relabelling_source(self):
        cache,shard=self.inputs()
        subset,records,traces=subset_inputs(cache,shard,list(range(128)))
        self.assertEqual(len(subset['batches']),2);self.assertEqual(len(records),128)
        self.assertEqual(subset['manifest_sha256'],'old');self.assertEqual(subset['result']['samples'],128)
        self.assertEqual(subset['source_result']['samples'],192)
        self.assertEqual(len(cache['batches']),3);self.assertEqual(len(cache['indices']),192)
        compare_pair(records,copy.deepcopy(records),traces,copy.deepcopy(traces))

    def test_partial_missing_reordered_inputs_are_rejected(self):
        cache,shard=self.inputs()
        for indices in (list(range(127)),list(range(256)),list(reversed(range(128)))):
            with self.assertRaises(RuntimeError):subset_inputs(cache,shard,indices)
        shard['batch_traces'][1]['global_start']=999
        with self.assertRaises(RuntimeError):subset_inputs(cache,shard,list(range(128)))

    def test_time_length_or_rng_changes_fail_but_category_changes_are_measured(self):
        cache,shard=self.inputs();_,before,traces=subset_inputs(cache,shard,list(range(128)))
        after=copy.deepcopy(before);after[0]['marks'][0]=5;after[0]['checkins'][0]=9
        compare_pair(before,after,traces,copy.deepcopy(traces))
        after[0]['arrival_times'][0]=.3
        with self.assertRaises(RuntimeError):compare_pair(before,after,traces,copy.deepcopy(traces))
        changed=copy.deepcopy(traces);changed[1]['spatial_rng_after']='changed'
        with self.assertRaises(RuntimeError):compare_pair(before,before,traces,changed)

    def test_delivery_hashes_and_zero_mismatch_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def save(name,value):(root/name).write_text(json.dumps(value),encoding='utf-8')
            save('manifest.json',{'revision':'ny-category-sampled-v2'})
            save('comparison.json',{'on':{'category_poi_mismatch_rate':0.0}})
            save('status.json',{'state':'complete'})
            files={n:sha256_file(root/n) for n in ('manifest.json','comparison.json')}
            save('audit.json',dict(state='passed',new_samples=128,baseline_samples=128,
                 originals_unchanged=True,paired_rng=True,files=files))
            self.assertEqual(verify_delivery(root)['actual_mismatch_rate'],0)
            save('comparison.json',{'on':{'category_poi_mismatch_rate':.2}})
            with self.assertRaises(RuntimeError):verify_delivery(root)


if __name__=='__main__':unittest.main()
