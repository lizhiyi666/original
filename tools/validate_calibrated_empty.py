"""Recreate the legacy empty temporal fixture, then validate both decoders in FP32.

TF32 is used ONLY to reconstruct the known historical input. It is switched off
before decoding. Full calibrated OOD resampling never enables TF32.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from datamodule import Batch
from evaluate_utils import get_task,get_run_data
from constraint_projection import ConstraintProjection
from experiment_io import (atomic_json,publish_torch,sha256_file,seed_sampling,
                           strict_test_matrix,decode_preserving_empty,validate_sequences)
from tools.perfcal_common import projector_kwargs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    source=Path(args.source_experiment)
    folder=Path(args.output_dir)
    folder.mkdir(exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    calibration=json.loads((Path(args.calibration)/'manifest.json').read_text())
    recommended=json.loads((Path(args.calibration)/'recommendation.json').read_text())['recommendation']
    expected=dict(checkpoint_sha256=sha256_file(source/'final.ckpt'),
                  projector_sha256=sha256_file(ROOT/'constraint_projection.py'),
                  validator_sha256=sha256_file(__file__),
                  calibration_manifest_sha256=sha256_file(Path(args.calibration)/'manifest.json'),
                  reference_indices=list(range(64,128)))
    receipt_path=folder/'receipt.json'
    if receipt_path.exists():
        saved=json.loads(receipt_path.read_text())
        if not args.resume or saved['inputs']!=expected or saved['state']!='passed':
            raise RuntimeError('Existing fixture proof requires matching completed --resume')
        for name,digest in saved['output_hashes'].items():
            if sha256_file(folder/name)!=digest:
                raise RuntimeError('Fixture proof artifact changed')
        return
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    _,_,run_path=get_run_data(manifest['run_id'],ROOT/'wandb')
    task,dm=get_task(run_path,str(ROOT),str(source/'final.ckpt'))
    raw=torch.load(ROOT/f'data/{manifest["dataset"]}/{manifest["dataset"]}_test.pkl',
                   map_location='cpu',weights_only=False)
    selected=dm.test_data.sequences[64:128]
    for index,sequence in enumerate(selected,64):
        sequence.po_matrix=strict_test_matrix(raw['sequences'][index],raw['poi_category'],raw['category_mapping'])
    batch=Batch.from_sequence_list(selected).to(task.device)
    torch.backends.cudnn.allow_tf32=True
    seed_sampling(135462)
    with torch.no_grad():
        temporal=task.tpp_model.sample(batch.batch_size,x_n=batch,tmax=batch.tmax)
    torch.backends.cudnn.allow_tf32=False
    temporal=temporal.mask_check()
    temporal.po_matrix=batch.po_matrix
    if int(temporal.unpadded_length[35])!=0:
        raise RuntimeError('Historical index-99 temporal fixture could not be reproduced')
    frozen=temporal.to('cpu')
    fixture_path=folder/'legacy-temporal-input.pkl'
    publish_torch(fixture_path,dict(batch=frozen,inputs=expected,fixture_generation_cudnn_tf32=True))
    dd=task.discrete_diffusion
    profile=calibration['profile']
    dd.projection_frequency=profile['projection']['projection_frequency']
    dd.projection_last_k_steps=profile['projection']['projection_last_k_steps']
    dd.constraint_projector=ConstraintProjection(**projector_kwargs(dd,profile,recommended['temperature']),
                                                 device=str(task.device),verbose=False)
    hashes={fixture_path.name:sha256_file(fixture_path)}
    calls={}
    for method in ('native','projection'):
        dd.use_constraint_projection=method=='projection'
        dd.projection_call_count=0
        seed_sampling(135462)
        decoded=decode_preserving_empty(task,copy.deepcopy(frozen).to(task.device),raw['poi_gps'])
        validate_sequences(decoded,raw['poi_category'])
        if len(decoded)!=64 or len(decoded[35]['checkins'])!=0:
            raise RuntimeError('Frozen mixed-batch index 99 was lost or filled')
        if (dd.projection_call_count>0)!=(method=='projection'):
            raise RuntimeError('Wrong projection switch in frozen-input regression')
        target=folder/f'{method}.pkl'
        publish_torch(target,dict(sequences=decoded,reference_indices=list(range(64,128))))
        hashes[target.name]=sha256_file(target)
        calls[method]=dd.projection_call_count
    atomic_json(receipt_path,dict(state='passed',inputs=expected,output_hashes=hashes,
        empty_index_99_preserved=True,decoder_precision='FP32; TF32 disabled',
        fixture_generation_cudnn_tf32=True,production_cudnn_tf32=False,projection_calls=calls))
    print('Frozen legacy empty input passed native and calibrated FP32 projection',flush=True)


if __name__=='__main__':
    main()
