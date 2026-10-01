"""Full-budget replay of the failing global test slice, without retraining."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiment_io import atomic_json,sha256_file,validate_sequences
from merge_results import merge_parts
from tools.run_newyork_ood import PROJECTION,DATASET,DATA_HASHES,code_fingerprint
from tools.continue_newyork_sampling import verify_compatible_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-experiment',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    source=Path(args.source_experiment).resolve()
    manifest=json.loads((source/'manifest.json').read_text())
    hashes=code_fingerprint()
    verify_compatible_sources(manifest['code_sha256'],hashes,'emptyfix-v1')
    checkpoint=source/'final.ckpt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['epoch']!=999:
        raise RuntimeError('Expected final 1000-epoch checkpoint')
    del saved
    folder=ROOT/'validation'
    folder.mkdir(exist_ok=True)
    receipt_path=folder/'emptyfix-validation.json'
    receipt=dict(state='running',sampling_revision='emptyfix-v1',sampling_code_sha256=hashes,
                 source_manifest_sha256=sha256_file(source/'manifest.json'),
                 checkpoint_sha256=sha256_file(checkpoint),projection=PROJECTION,
                 effective_batch_seed=135462,regression_test_indices=list(range(64,128)),started_at=time.time())
    atomic_json(receipt_path,receipt)
    outputs={}
    try:
        with (folder/'unit-tests.log').open('w') as log:
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        for method in ('native','projection'):
            tag=f"{manifest['run_id']}_emptyfix-v1_regression_{method}"
            command=[sys.executable,'-B','sample.py','--run_id',manifest['run_id'],
                     '--checkpoint',str(checkpoint),'--output_tag',tag,'--seed','135398',
                     '--batch_size','64','--start_index','64','--max_samples','64',
                     '--world_size','1','--constraint_source','strict_test','--sampling_revision','emptyfix-v1']
            if args.resume:
                command+=['--resume']
            if method=='projection':
                command+=['--use_constraint_projection','--use_gumbel_softmax']
                for key,value in PROJECTION.items():
                    command+=[f'--{key}',str(value)]
            print(f'Replaying batch 64..127: {method}',flush=True)
            started=time.monotonic()
            with (folder/f'regression-{method}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            path=merge_parts(DATASET,manifest['run_id'],1,tag,expected_count=64)
            data=torch.load(path,map_location='cpu',weights_only=False)
            if data['test_indices']!=list(range(64,128)) or len(data['sequences'])!=64:
                raise RuntimeError('Regression index/count mismatch')
            if 99 not in data['temporal_empty_test_indices'] or len(data['sequences'][35]['checkins'])!=0:
                raise RuntimeError('Known empty test index 99 was not preserved')
            if method=='projection' and data['projection_calls']<=0:
                raise RuntimeError('Projection did not execute')
            if method=='native' and data['projection_calls']!=0:
                raise RuntimeError('Native path used projection')
            validate_sequences(data['sequences'])
            outputs[method]=data
            receipt[f'{method}_seconds']=time.monotonic()-started
            receipt[f'{method}_output_sha256']=sha256_file(path)
            atomic_json(receipt_path,receipt)
        if outputs['native']['temporal_empty_test_indices']!=outputs['projection']['temporal_empty_test_indices']:
            raise RuntimeError('Paired methods have different temporal-empty indices')
        for split,expected in DATA_HASHES.items():
            relative=f'data/{DATASET}/{DATASET}_{split}.pkl'
            if any(sha256_file(root/relative)!=expected for root in (ROOT,source.parent.parent)):
                raise RuntimeError('Input dataset changed')
        receipt.update(state='passed',empty_index_99_preserved=True,
                       projection_calls=outputs['projection']['projection_calls'],completed_at=time.time())
    except BaseException as exc:
        receipt.update(state='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        atomic_json(receipt_path,receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
