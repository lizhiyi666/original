"""Shared, fail-closed I/O for baseline-suite-v1."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf
from configs import instantiate_model
from evaluate_utils import get_task, get_run_data
from experiment_io import sha256_file, strict_test_matrix, validate_sequences
from tools.perfcal_common import calibration_metrics
from evaluations.ovr import dataset_unsat_ratio_by_test_pairs
from evaluations.statistical_metrics import Get_Statistical_Metrics

ROOT=Path(__file__).resolve().parents[1]
DATASET='NewYork_PO1_OOD'
SEED=135398
CFG_FORMAT='baseline-suite-cfg-v1'


def precision():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)


def state_hash(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape,str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_checkpoint(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False,suffix='.tmp') as stream:
        temporary=Path(stream.name)
        torch.save(value,stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def fresh_cfg(task,dm,run_path):
    config=OmegaConf.load(Path(run_path)/'config_hydra.yaml')
    config.model.po_cfg_enabled=True
    config.model.use_constraint_projection=False
    unused_time,spatial=instantiate_model(config.model,dm)
    del unused_time
    device=task.device
    task.discrete_diffusion=spatial.to(device)
    task.tpp_model.requires_grad_(False)
    task.tpp_model.eval()
    return spatial


def install_cfg(task,dm,run_path,path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('format')!=CFG_FORMAT or saved.get('trained') is not True
            or saved.get('epoch')!=999 or saved.get('global_step')!=50000
            or min(saved['branch_counts']['conditional_rows'],saved['branch_counts']['null_rows'])<=0):
        raise RuntimeError('CFG requires a completed 1000-epoch conditional/null checkpoint')
    if state_hash(task.tpp_model)!=saved['temporal_sha256']:
        raise RuntimeError('CFG temporal model differs from baseline1')
    dd=fresh_cfg(task,dm,run_path)
    dd.load_state_dict(saved['spatial_state_dict'],strict=True)
    dd.cfg_trained=True
    dd.eval()
    dd.reset_cfg_stats()
    return dd


def load_task(source,cfg_checkpoint=None):
    source=Path(source)
    manifest=json.loads((source/'manifest.json').read_text())
    _,_,run_path=get_run_data(manifest['run_id'],ROOT/'wandb')
    task,dm=get_task(run_path,str(ROOT),str(source/'final.ckpt'))
    if cfg_checkpoint:
        install_cfg(task,dm,run_path,cfg_checkpoint)
    return task,dm,run_path


def references(dm,split,indices):
    raw=torch.load(ROOT/f'data/{DATASET}/{DATASET}_{split}.pkl',map_location='cpu',weights_only=False)
    pool=getattr(dm,split+'_data').sequences
    selected=[pool[i] for i in indices]
    refs=[raw['sequences'][i] for i in indices]
    for sequence,reference in zip(selected,refs):
        sequence.po_matrix=strict_test_matrix(reference,raw['poi_category'],raw['category_mapping'])
    return raw,selected,refs


def validate_alignment(generated,refs,poi_category):
    if len(generated)!=len(refs):
        raise ValueError('Missing generated samples')
    validate_sequences(generated,poi_category)
    for reference,sample in zip(refs,generated):
        for i in range(1,7):
            key=f'condition{i}_indicator'
            if not np.array_equal(reference[key],sample[key]):
                raise ValueError(f'Condition/index mismatch: {key}')


def full_metrics(refs,generated,poi_category):
    metrics=calibration_metrics(refs,generated,poi_category)
    revised=[]
    for sample in generated:
        copy=dict(sample)
        copy['marks']=[poi_category[int(p)] for p in sample['checkins']]
        revised.append(copy)
    metrics.update({str(k):float(v) for k,v in Get_Statistical_Metrics(refs,revised).items()})
    metrics['Unsat_ref']=float(dataset_unsat_ratio_by_test_pairs(refs,generated,poi_category,skip_nan=True))
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Non-finite baseline evaluation')
    return metrics


def select_candidate(rows,method):
    valid=[r for r in rows if r['state']=='complete' and r.get('method_executed') is True
           and r.get('memory_fraction',1)<=.8]
    if method=='baseline3':
        valid=[r for r in valid if not r['all_violation_gradients_zero']]
    else:
        valid=[r for r in valid if r['scale']!=0 and r['branch_difference']>0]
    if not valid:
        raise RuntimeError(f'No valid {method} calibration candidate; no native fallback')
    return min(valid,key=lambda r:(r['metrics']['strict_ovr'],-r['metrics']['pair_coverage'],
                                   r['scale'],r.get('temperature',0)))


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
