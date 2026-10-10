import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment_io import sha256_file
from tools.category_cache_audit import (REVISION,EXPECTED_ERROR,audit_alignment,audit_traces,
    cache_matches_original,metrics_equal,verify_audit,verify_snapshot)
from tools.audit_category_decoding_cache import extract_snapshot


def fixture():
    lengths=[2,3,0];batch=types.SimpleNamespace(batch_size=3,unpadded_length=torch.tensor(lengths))
    batch.mask=torch.tensor([[1,1,0],[1,1,1],[0,0,0]],dtype=torch.bool)
    batch.time=torch.tensor([[1.,2.,0.],[3.,4.,5.],[0.,0.,0.]])
    mapping={7:4,8:5,9:4};gps={7:'0.0,1.0',8:'2.0,3.0',9:'4.0,5.0'}
    for n in range(1,7):
        setattr(batch,f'condition{n}',torch.arange(9).reshape(3,3)+n)
        setattr(batch,f'condition{n}_indicator',torch.tensor([[n,0],[0,n],[n,n]]))
    on=[];off=[]
    for i,n in enumerate(lengths):
        pois=[7,8,9][:n]
        row=dict(arrival_times=batch.time[i][batch.mask[i]].numpy(),checkins=np.array(pois,dtype=np.int64),
            marks=np.array([mapping[p] for p in pois]),gps=[[float(v) for v in gps[p].split(',')] for p in pois])
        for k in range(1,7):
            row[f'condition{k}']=getattr(batch,f'condition{k}')[i][batch.mask[i]].numpy()
            row[f'condition{k}_indicator']=getattr(batch,f'condition{k}_indicator')[i].numpy()
        on.append(row);old=copy.deepcopy(row)
        keep=([1],[0,2],[])[i]
        for key in ('arrival_times','checkins','marks','gps')+tuple(f'condition{k}' for k in range(1,7)):
            v=np.asarray(old[key])[keep]
            old[key]=v.tolist() if key=='gps' else v.copy()
        off.append(old)
    return [dict(indices=[0,1,2],global_start=0,batch=batch)],off,on,mapping,gps


class CategoryCacheAuditTests(unittest.TestCase):
    def test_recovered_events_pass_cache_alignment_without_rewriting_outputs(self):
        items,off,on,mapping,gps=fixture();before=copy.deepcopy(on)
        rng=torch.get_rng_state().clone()
        with patch('torch.rand',side_effect=AssertionError('Audits must not sample')):
            result,matched=audit_alignment(items,off,on,mapping,gps)
        self.assertEqual(result['restored_events'],2)
        self.assertEqual([r['index'] for r in result['restored_rows']],[0,1])
        self.assertEqual(result['off_events'],3);self.assertEqual(result['on_events'],5)
        self.assertEqual([len(r['checkins']) for r in matched],[1,2,0])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for a,b in zip(before,on):
            for key in a:self.assertTrue(np.array_equal(a[key],b[key]))

    def test_on_time_or_condition_change_fails_even_with_equal_lengths(self):
        for key in ('arrival_times','condition3','condition1_indicator'):
            items,off,on,mapping,gps=fixture();on=copy.deepcopy(on);on[0][key][0]+=1
            with self.assertRaises(RuntimeError):audit_alignment(items,off,on,mapping,gps)

    def test_dropping_on_event_does_not_pass_under_new_basis(self):
        items,off,on,mapping,gps=fixture();on[0]['checkins']=on[0]['checkins'][:1]
        with self.assertRaisesRegex(RuntimeError,'retain every cached event'):audit_alignment(items,off,on,mapping,gps)

    def test_off_must_be_an_ordered_exact_subset(self):
        items,off,on,mapping,gps=fixture();off[1]['arrival_times']=off[1]['arrival_times'][::-1]
        with self.assertRaisesRegex(RuntimeError,'ordered subset'):audit_alignment(items,off,on,mapping,gps)
        items,off,on,mapping,gps=fixture();off[0]['condition2'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'Off conditions'):audit_alignment(items,off,on,mapping,gps)

    def test_actual_category_and_catalog_gps_are_required(self):
        items,off,on,mapping,gps=fixture();on[0]['marks'][0]=5
        with self.assertRaisesRegex(RuntimeError,'Actual sampled category'):audit_alignment(items,off,on,mapping,gps)
        items,off,on,mapping,gps=fixture();on[0]['gps'][0][0]=999.
        with self.assertRaisesRegex(RuntimeError,'GPS differs'):audit_alignment(items,off,on,mapping,gps)

    def test_every_cache_attribute_must_match_original(self):
        items,*_=fixture();original=copy.deepcopy(items)
        cache_matches_original(items,original)
        original[0]['batch'].mask[0,0]=False
        with self.assertRaisesRegex(RuntimeError,'original cache'):cache_matches_original(items,original)

    def test_rng_and_budget_cannot_be_relaxed(self):
        items,*_=fixture()
        trace=dict(indices=[0,1,2],global_start=0,spatial_rng_before='s0',spatial_rng_after='s1',
            projection_rng_after='p1',distance_rng_before='d0',distance_rng_after='d1',
            global_rng_unchanged=True,effective_constraints=True,
            projection_stats=[dict(diffusion_step=s,optimizer_steps=500) for s in range(36,-1,-4)])
        audit_traces(items,[trace],[copy.deepcopy(trace)])
        for key,value in (('spatial_rng_after','changed'),('global_rng_unchanged',False)):
            changed=copy.deepcopy(trace);changed[key]=value
            with self.assertRaises(RuntimeError):audit_traces(items,[trace],[changed])
        changed=copy.deepcopy(trace);changed['projection_stats'][0]['optimizer_steps']=499
        with self.assertRaisesRegex(RuntimeError,'budget'):audit_traces(items,[trace],[changed])

    def test_stored_metrics_must_be_reproduced_including_undefined_values(self):
        metrics_equal({'x':.2,'y':None},{'x':.2,'y':None})
        for stored in ({'x':.21,'y':None},{'x':float('nan'),'y':None},{'x':.2,'y':0.}):
            with self.assertRaises(RuntimeError):metrics_equal({'x':.2,'y':None},stored)

    def test_snapshot_transport_and_original_failed_state_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);status=json.dumps(dict(state='failed',error=EXPECTED_ERROR)).encode()
            contents={'source/run/status.json':status}
            seal=dict(state='sealed',source_state='failed',source_manifest_sha256='source',
                      files={k:hashlib.sha256(v).hexdigest() for k,v in contents.items()})
            inventory=json.dumps(seal).encode();contents['snapshot-inventory.json']=inventory
            path=root/'snapshot.tar.gz'
            with tarfile.open(path,'w:gz') as tar:
                for name,value in contents.items():
                    entry=tarfile.TarInfo(name);entry.size=len(value);tar.addfile(entry,io.BytesIO(value))
            server=dict(sha256=sha256_file(path),inventory_sha256=hashlib.sha256(inventory).hexdigest())
            extract_snapshot(path,root/'out',server)
            self.assertEqual(verify_snapshot(root/'out')['source_state'],'failed')
            (root/'out/source/run/status.json').write_text(json.dumps({'state':'complete'}),encoding='utf-8')
            with self.assertRaises(RuntimeError):verify_snapshot(root/'out')

    def test_snapshot_symlink_and_escape_rejected_before_writing(self):
        for name,kind in (('link','symlink'),('../escape','regular')):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);path=root/'unsafe.tar.gz'
                with tarfile.open(path,'w:gz') as tar:
                    entry=tarfile.TarInfo(name)
                    if kind=='symlink':entry.type=tarfile.SYMTYPE;entry.linkname='/etc/passwd'
                    tar.addfile(entry)
                with self.assertRaises(RuntimeError):extract_snapshot(path,root/'out',dict(sha256=sha256_file(path)))
                self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
