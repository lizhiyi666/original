"""Dataset-aware adapters. Preserve legacy model slots without relabeling semantic data."""
import json
import math
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf
from configs import instantiate_datamodule,instantiate_model,instantiate_task
from experiment_io import sha256_file,strict_test_matrix,atomic_json
from tools.baseline_common import state_hash,CFG_FORMAT

ROOT=Path(__file__).resolve().parents[1]
APPROVED={
    'NewYork_PO1_OOD': dict(train_count=3160,test_count=2108,semantic_classes=9,model_category_slots=9,
        legacy_train_batch=64,checkpoint_sha256='eb3c765d367cb507d427e6f0e6a33f2c5c2ee2d6d0bf9c742b31e4d1b0907ba9',
        data_sha256={'train':'afdadf2d60329afc5dfb86aa792f5a359455f8de720eab8d56e23c376593afec',
                     'test':'6f127b1bb0562500e1e679ac564fdaab778ccb76fecc7087be63f6590df077ab'}),
    'Istanbul_PO1_OOD': dict(train_count=7035,test_count=4914,semantic_classes=9,model_category_slots=10,
        legacy_train_batch=512,checkpoint_sha256='a20566fcbd7980fb22e265700c81e3dd891ae60573e4c1c0232e41b1ebe334ba',
        data_sha256={'train':'736913a243ee2cc9aebd2ec5d0b2182cb025f8c0b649ea78aca5f21d701ba4d8',
                     'test':'c4c8262709eed414a2719883d2849855e438cbacf566f43f443e7af8abedb10b'}),
}


def cfg_steps(profile,epochs=1000):
    return math.ceil(profile['train_count']/64)*epochs


def profile(name,checkpoint,config_path,data_root):
    p=dict(APPROVED[name],dataset=name,checkpoint=str(Path(checkpoint).resolve()),
           config_path=str(Path(config_path).resolve()),data_root=str(Path(data_root).resolve()))
    if sha256_file(p['checkpoint'])!=p['checkpoint_sha256']:
        raise RuntimeError(f'Unapproved {name} checkpoint')
    p['config_sha256']=sha256_file(p['config_path'])
    cfg=OmegaConf.load(p['config_path'])
    if cfg.data.name!=name or int(cfg.data.batch_size)!=p['legacy_train_batch']:
        raise RuntimeError('Historical checkpoint config differs from approved source')
    maps=[]
    for split in ('train','test'):
        path=Path(p['data_root'])/name/f'{name}_{split}.pkl'
        if sha256_file(path)!=p['data_sha256'][split]:
            raise RuntimeError(f'Original {name} {split} bytes changed')
        raw=torch.load(path,map_location='cpu',weights_only=False)
        mapping={int(k):int(v) for k,v in raw['category_mapping'].items()}
        if (len(raw['sequences'])!=p[split+'_count'] or len(mapping)!=p['semantic_classes']
                or int(raw['num_marks'])!=p['model_category_slots']
                or sorted(mapping.values())!=list(range(p['semantic_classes']))):
            raise RuntimeError('Dataset sizes/category mapping changed')
        maps.append(mapping)
    if maps[0]!=maps[1]:
        raise RuntimeError('Train/test category mappings differ')
    p['category_mapping']={str(k):v for k,v in maps[0].items()}
    p['cfg_expected_steps']=cfg_steps(p)
    p['legacy_training_data_fingerprint_available']=name=='NewYork_PO1_OOD'
    return p


def expanded_matrix(matrix,semantic_classes,model_slots):
    if matrix.shape!=(semantic_classes,semantic_classes) or model_slots<semantic_classes:
        raise ValueError('Invalid semantic/model-slot dimensions')
    if model_slots==semantic_classes:
        return matrix
    result=matrix.new_zeros(model_slots,model_slots)
    result[:semantic_classes,:semantic_classes]=matrix
    return result


def references(dm,p,split,indices):
    path=Path(p['data_root'])/p['dataset']/f"{p['dataset']}_{split}.pkl"
    if sha256_file(path)!=p['data_sha256'][split]:
        raise RuntimeError('Dataset changed after manifest was frozen')
    raw=torch.load(path,map_location='cpu',weights_only=False)
    sequences=getattr(dm,split+'_data').sequences
    selected=[sequences[i] for i in indices]
    refs=[raw['sequences'][i] for i in indices]
    for seq,ref in zip(selected,refs):
        seq.po_matrix=expanded_matrix(strict_test_matrix(ref,raw['poi_category'],raw['category_mapping']),
                                      p['semantic_classes'],p['model_category_slots'])
    return raw,selected,refs


def source_config(p):
    if sha256_file(p['config_path'])!=p['config_sha256']:
        raise RuntimeError('Source config changed')
    config=OmegaConf.load(p['config_path'])
    config.data.root=p['data_root']
    config.data.batch_size=64
    config.model.use_constraint_projection=False
    config.model.po_cfg_enabled=False
    # Inference and CFG spatial-only training never invoke the historical auxiliary task loss.
    config.task.po_loss_weight=0
    return config


def load_city(p,cfg_checkpoint=None,device=None):
    if sha256_file(p['checkpoint'])!=p['checkpoint_sha256']:
        raise RuntimeError('Base weights changed')
    config=source_config(p)
    from add_thin.utils.seed import set_seed
    set_seed(config)
    dm=instantiate_datamodule(config.data); dm.prepare_data()
    if (len(dm.train_data)!=p['train_count'] or len(dm.test_data)!=p['test_count']
            or dm.num_category!=p['model_category_slots']):
        raise RuntimeError('Loaded dataset does not match profile')
    tpp,dd=instantiate_model(config.model,dm)
    task=instantiate_task(config.task,tpp,dd,dm)
    saved=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)
    if saved.get('epoch')!=999 or 'state_dict' not in saved:
        raise RuntimeError('Expected completed original base model')
    task.load_state_dict(saved['state_dict'],strict=True)
    del saved
    task.to(device or ('cuda' if torch.cuda.is_available() else 'cpu')).eval()
    if cfg_checkpoint:
        install_city_cfg(task,dm,p,cfg_checkpoint)
    return task,dm


def fresh_city_cfg(task,dm,p):
    config=source_config(p)
    config.model.po_cfg_enabled=True
    unused,dd=instantiate_model(config.model,dm)
    del unused
    task.discrete_diffusion=dd.to(task.device)
    task.tpp_model.requires_grad_(False).eval()
    return dd


def install_city_cfg(task,dm,p,path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('format')!=CFG_FORMAT or saved.get('trained') is not True
            or saved.get('epoch')!=999 or saved.get('global_step')!=cfg_steps(p)
            or min(saved['branch_counts']['conditional_rows'],saved['branch_counts']['null_rows'])<=0):
        raise RuntimeError('CFG capability/update budget mismatch')
    if state_hash(task.tpp_model)!=saved['temporal_sha256']:
        raise RuntimeError('CFG would replace the frozen temporal model')
    if saved['job'].get('city') and saved['job']['city']!=p:
        raise RuntimeError('CFG was trained for another dataset/profile')
    dd=fresh_city_cfg(task,dm,p)
    dd.load_state_dict(saved['spatial_state_dict'],strict=True)
    dd.cfg_trained=True; dd.cfg_scale=1.0; dd.eval(); dd.reset_cfg_stats()
    return dd


def verify_tracking(entity,run_id,expected,receipt,url):
    import wandb
    last=None
    for delay in (0,2,5):
        if delay: time.sleep(delay)
        try:
            remote=wandb.Api(timeout=30).run(f'{entity}/Marionette/{run_id}')
            if all(remote.summary.get(k)==v for k,v in expected.items()):
                atomic_json(receipt,dict(state='verified',expected=expected,url=url))
                return
            last='Remote summary is not consistent yet'
        except Exception as exc:
            last=str(exc)
    raise RuntimeError(f'Tracking not confirmed; numerical outputs retained, resume sync only: {last}')
