"""Locked protocol and diagnostics for the first PCDG ablation study."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from constraint_projection import ConstraintProjection
from evaluations.ovr import (_seq_cats_order, _extract_reference_pairs_for_sequence,
                             _violation_rate_for_pair_in_generated)
from evaluations.statistical_metrics import Get_Statistical_Metrics, EVALUATION_VERSION
from tools.baseline_common import validate_alignment

VERSION='pcdg-ablation-v1'
SEEDS=(135398,135399,135400)
VARIANTS={
    'full': dict(order=1,existence=5,kl=1,update=True,gumbel=True),
    'no_projection': None,
    'no_existence': dict(order=1,existence=0,kl=1,update=True,gumbel=True),
    'no_order': dict(order=0,existence=5,kl=1,update=True,gumbel=True),
    'no_kl': dict(order=1,existence=5,kl=0,update=True,gumbel=True),
    'fixed_multipliers': dict(order=1,existence=5,kl=1,update=False,gumbel=True),
    'no_gumbel': dict(order=1,existence=5,kl=1,update=True,gumbel=False),
}
DISTANCE_VERSION = 'pcdg-distance-v2'
DISTANCE_VARIANTS = {name: (None if value is None else dict(value, distance=0 if name == 'no_kl' else 1))
                     for name, value in VARIANTS.items()}
DISTANCE_VARIANTS['no_distance_kl'] = dict(DISTANCE_VARIANTS['full'], distance=0)
STEPS=list(range(36,-1,-4))


def stream_seed(seed,global_start,stream):
    if stream not in ('spatial','projection'):
        raise ValueError('Unknown random stream')
    key=json.dumps([VERSION,int(seed),int(global_start),stream],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big') & ((1<<63)-1)


def generator(seed,global_start,stream,device):
    return torch.Generator(device=device).manual_seed(stream_seed(seed,global_start,stream))


def rng_digest(value):
    state=value.get_state() if isinstance(value,torch.Generator) else value
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def projector(dd,variant,rng=None,*,revision=VERSION,datamodule=None,distance_backend='legacy'):
    if revision not in (VERSION, DISTANCE_VERSION):
        raise ValueError('Unknown projection revision')
    p=(DISTANCE_VARIANTS if revision == DISTANCE_VERSION else VARIANTS)[variant]
    if p is None:
        return None
    instance = ConstraintProjection(dd.num_classes,dd.type_classes,dd.num_spectial,
        tau=0,lambda_init=1,mu_init=1,mu_alpha=2,mu_max=1000,
        outer_iterations=10,inner_iterations=50,eta=1,delta_tol=1e-6,
        use_gumbel_softmax=p['gumbel'],gumbel_temperature=3,
        projection_order_weight=p['order'],projection_existence_weight=p['existence'],
        projection_kl_weight=p['kl'],update_multipliers=p['update'],early_stop=False,
        generator=rng,collect_diagnostics=True,verbose=False,
        projection_distance_kl_weight=p.get('distance', 0), distance_backend=distance_backend)
    if instance.projection_distance_kl_weight:
        if datamodule is None:
            raise ValueError('Distance ablations require the training datamodule')
        from distance_kl import attach_distance_reference
        attach_distance_reference(instance, datamodule)
    return instance


def _rates(pairs,cats):
    missing,wrong,present,unsat=0,0.0,0,0
    skip=[]
    for a,b in pairs:
        v,n=_violation_rate_for_pair_in_generated(cats,a,b)
        if n==0:
            missing+=1
            unsat+=1
        else:
            present+=1
            wrong+=v/n
            skip.append(v/n)
            unsat+=int(v>0)
    k=len(pairs)
    return dict(reference_pairs=k,present_pairs=present,unsatisfied_pairs=unsat,
                missing_contribution=missing/k if k else None,
                order_contribution=wrong/k if k else None,
                strict_ovr=(missing+wrong)/k if k else None,
                ovr_skip=float(np.mean(skip)) if skip else None)


def evaluate(refs,generated,poi_category,indices,*,diagnostics=None):
    validate_alignment(generated,refs,poi_category)
    if len(indices)!=len(generated) or len(set(indices))!=len(indices):
        raise ValueError('Invalid evaluation indices')
    rows=[]
    for index,ref,gen in zip(indices,refs,generated):
        pairs=_extract_reference_pairs_for_sequence(_seq_cats_order(ref,poi_category))
        cats=_seq_cats_order(gen,poi_category)
        row=dict(index=index,**_rates(pairs,cats))
        involved={c for pair in pairs for c in pair}
        row.update(involved_categories=len(involved),covered_categories=len(involved.intersection(cats)),
                   events=len(cats),empty=len(cats)==0,
                   category_poi_mismatches=sum(int(a)!=int(b) for a,b in zip(gen['marks'],cats)),
                   token_strict_ovr=_rates(pairs,[int(c) for c in gen['marks']])['strict_ovr'])
        if len(gen['marks'])!=len(cats):
            raise ValueError('Raw category tokens lost alignment with POIs')
        rows.append(row)
    def mean(key):
        values=[r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    pairs=sum(r['reference_pairs'] for r in rows)
    involved=sum(r['involved_categories'] for r in rows)
    events=sum(r['events'] for r in rows)
    metrics={k:mean(k) for k in ('strict_ovr','ovr_skip','missing_contribution','order_contribution','token_strict_ovr')}
    metrics.update(pair_coverage=sum(r['present_pairs'] for r in rows)/pairs if pairs else None,
        category_coverage=sum(r['covered_categories'] for r in rows)/involved if involved else None,
        Unsat=sum(r['unsatisfied_pairs'] for r in rows)/pairs if pairs else None,
        category_poi_mismatch_rate=sum(r['category_poi_mismatches'] for r in rows)/events if events else None,
        empty_count=sum(r['empty'] for r in rows),empty_rate=sum(r['empty'] for r in rows)/len(rows),
        ovr_valid_conditions=sum(r['strict_ovr'] is not None for r in rows),
        ovr_skip_valid_conditions=sum(r['ovr_skip'] is not None for r in rows))
    if metrics['strict_ovr'] is not None and abs(metrics['strict_ovr']-
            metrics['missing_contribution']-metrics['order_contribution'])>1e-12:
        raise RuntimeError('Strict OVR decomposition mismatch')
    revised=[dict(s,marks=[poi_category[int(p)] for p in s['checkins']]) for s in generated]
    stats=Get_Statistical_Metrics(refs,revised,diagnostics=diagnostics)
    metrics.update({str(k):float(v) if math.isfinite(float(v)) else None for k,v in stats.items()})
    metrics['evaluation_version'] = EVALUATION_VERSION
    return metrics,rows


def same_records(left,right):
    return len(left)==len(right) and all(a.keys()==b.keys() and
        all(np.array_equal(a[k],b[k]) for k in a) for a,b in zip(left,right))


def mean_sd(values):
    if any(v is None for v in values):
        return dict(mean=None,sample_sd=None,n=len(values))
    return dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                n=len(values))
