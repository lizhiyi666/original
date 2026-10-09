"""Read-only remote audit and additive, hash-checked local experiment delivery.

Never imports training code, unpickles outputs, or overwrites source artifacts.
Derived reports are kept separately from immutable experiment metadata.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'experiment_runs'
OUT = RUNS / 'integrated-20261005'
STAGE = ROOT / 'tmp' / 'two-city-v1-server'
REMOTE = '/root/experiments/pcdg/two-city-v1/experiment_runs/two-city-v1'
LEGACY = 'nyood1000s13539820261001_perfcal-v1-ood-r2'
BASELINES = 'nyood-baselines-v1-20261002'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    """Write only derived integration files, never source artifacts."""
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError(f'Not an integration output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


REMOTE_PROBE = r'''
import json, hashlib
from pathlib import Path
from collections import Counter
root=Path('/root/experiments/pcdg/two-city-v1/experiment_runs/two-city-v1')
project=root.parent.parent
def read(p): return json.loads(Path(p).read_text())
memo={}
def sha(p):
 p=Path(p)
 if str(p) not in memo:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''): h.update(b)
  memo[str(p)]=h.hexdigest()
 return memo[str(p)]
errors=[]; checked=Counter()
def check(p,expected,kind):
 try:
  actual=sha(p)
  if actual!=expected: errors.append(dict(path=str(p),kind=kind,expected=expected,actual=actual))
  checked[kind]+=1
 except Exception as e: errors.append(dict(path=str(p),kind=kind,error=str(e)))
manifest=read(root/'manifest.json'); registry=read(root/'registry.json')
recovery=read(root/'recovery/empty-fixture-v1/manifest.json')
for k,r in registry.items():
 check(r['path'],r['sha256'],'results')
 for s in r['source_shards']: check(s['path'],s['sha256'],'source_shards')
 if r['origin']=='new':
  receipt=read(Path(r['path']).parent/'wandb-sync.json')
  if receipt.get('state')!='verified' or receipt.get('expected',{}).get('result_sha256')!=r['sha256']: errors.append(dict(receipt=k))
  else: checked['wandb_receipts']+=1
for p,h in recovery['preserved_sha256'].items(): check(p,h,'preserved')
for p,h in recovery['recovery_code_sha256'].items(): check(p,h,'recovery_code')
check(root/'manifest.json',recovery['source_manifest_sha256'],'original_manifest')
for p,h in manifest['code_sha256'].items(): check(project/p,h,'production_code')
check(Path(manifest['source_ny_ablation'])/'manifest.json',manifest['source_ny_manifest_sha256'],'source_manifest')
check(manifest['newyork_cfg'],manifest['newyork_cfg_sha256'],'cfg_checkpoint')
for name,p in manifest['profiles'].items():
 check(p['checkpoint'],p['checkpoint_sha256'],'base_checkpoint')
 check(p['config_path'],p['config_sha256'],'config')
 for split,h in p['data_sha256'].items(): check(Path(p['data_root'])/name/(name+'_'+split+'.pkl'),h,'dataset')
cfg=root/'Istanbul_PO1_OOD/cfg-training'
check(cfg/'last.ckpt',read(cfg/'result.json')['checkpoint_sha256'],'cfg_checkpoint')
inventory={}
for p in sorted(root.rglob('*')):
 if p.is_file(): inventory[p.relative_to(root).as_posix()]=dict(size=p.stat().st_size,sha256=sha(p))
print(json.dumps(dict(state='passed' if not errors else 'failed',checked=dict(checked),errors=errors,
 counts=dict(Counter(r['dataset'] for r in registry.values())),status=read(root/'status.json'),
 main_audit=read(root/'audit.json'),recovery_audit=read(root/'recovery/empty-fixture-v1/audit.json'),inventory=inventory)))
'''


def probe():
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'pcdg',
                           '/root/anaconda3/envs/pcdg-exp/bin/python', '-B', '-'],
                          input=REMOTE_PROBE, text=True, encoding='utf-8', capture_output=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    evidence = json.loads(proc.stdout)
    evidence['verified_at'] = datetime.now(timezone.utc).isoformat()
    save(OUT / 'server-audit.json', evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k!='inventory'}, ensure_ascii=False), flush=True)
    print(f"Remote inventory: {len(evidence['inventory'])} files, {sum(x['size'] for x in evidence['inventory'].values())} bytes", flush=True)
    if evidence['state'] != 'passed':
        raise RuntimeError('Remote audit failed; no delivery attempted')


def safe_child(root, relative):
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe relative path: {relative}')
    return target


def copy_no_overwrite(source, dest, expected):
    if dest.exists():
        if digest(dest) != expected:
            raise RuntimeError(f'Existing destination has different content: {dest}')
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a file another process created after our check.
    with source.open('rb') as src, dest.open('xb') as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    if digest(dest) != expected:
        raise RuntimeError(f'Copy verification failed: {dest}')
    return True


def deliver():
    evidence = read(OUT / 'server-audit.json')
    assert evidence['state'] == 'passed'
    issues = []
    for rel, meta in evidence['inventory'].items():
        p = safe_child(STAGE, rel)
        if not p.is_file() or p.stat().st_size != meta['size'] or digest(p) != meta['sha256']:
            issues.append(rel)
    save(OUT / 'staging-audit.json', dict(state='passed' if not issues else 'failed',
                                        checked=len(evidence['inventory']), issues=issues))
    if issues:
        print(json.dumps(issues, ensure_ascii=False), flush=True)
        raise RuntimeError('Incomplete or changed staging files; do not merge')
    original = RUNS / 'two-city-v1'
    existing = {p.relative_to(original).as_posix(): digest(p) for p in original.rglob('*') if p.is_file()}
    entries = []
    for rel, meta in evidence['inventory'].items():
        source = safe_child(STAGE, rel)
        target = safe_child(original, rel)
        if target.exists() and digest(target) != meta['sha256']:
            target = safe_child(OUT / 'sources' / 'two-city-v1', rel)
            action = 'versioned_conflict'
        else:
            action = 'same_hash' if target.exists() else 'added'
        copy_no_overwrite(source, target, meta['sha256'])
        entries.append(dict(relative_path=rel, local_path=target.as_posix(), action=action, **meta))
    for rel, h in existing.items():
        if digest(original / rel) != h:
            raise RuntimeError(f'Pre-existing local file changed: {rel}')
    # A self-contained small metadata snapshot makes the final report discoverable,
    # while large generated files are stored only in their canonical run directories.
    for rel, meta in evidence['inventory'].items():
        if '/' not in rel and Path(rel).suffix in ('.md', '.json', '.csv', '.tex'):
            copy_no_overwrite(STAGE / rel, OUT / 'sources' / 'two-city-v1' / rel, meta['sha256'])
    save(OUT / 'merge-audit.json', dict(state='passed', preserved_existing_files=len(existing),
                                     counts=dict(Counter(e['action'] for e in entries)), files=entries))
    print(json.dumps(dict(preserved_existing_files=len(existing), counts=dict(Counter(e['action'] for e in entries)))), flush=True)


def local_path(remote):
    marker='/experiment_runs/'
    if marker not in remote:
        raise ValueError(remote)
    return safe_child(RUNS, remote.split(marker, 1)[1])


def audit_local():
    records=read(OUT/'sources/two-city-v1/registry.json')
    index=[]; errors=[]; checked=Counter()
    for key,r in records.items():
        p=local_path(r['path'])
        for path,expected,kind in [(p,r['sha256'],'results')]+[(local_path(s['path']),s['sha256'],'source_shards') for s in r['source_shards']]:
            if not path.exists() or digest(path)!=expected: errors.append(str(path))
            else: checked[kind]+=1
        index.append(dict(id=key,dataset=r['dataset'],seed=r['seed'],variant=r['variant'],origin=r['origin'],
                          local_path=p.as_posix(),sha256=r['sha256'],metrics=r['metrics'],
                          source_shards=[dict(local_path=local_path(s['path']).as_posix(),sha256=s['sha256']) for s in r['source_shards']]))
    comparison=read(RUNS/BASELINES/'comparison.json')
    for method,r in comparison.items():
        if method in ('baseline1','projection'):
            p=RUNS/LEGACY/'generated'/Path(r['output']).name
        else: p=RUNS/BASELINES/(method+'.pkl')
        if not p.exists() or digest(p)!=r['output_sha256']: errors.append(str(p))
        else: checked['historical_unique_results']+=1
        index.append(dict(id='historical/'+method,dataset='NewYork_PO1_OOD',seed=135398,variant=method,
                          origin='historical_single_seed',local_path=p.as_posix(),sha256=r['output_sha256'],metrics=r['metrics']))
    # Independently recompute all published means/sample SDs from registry values.
    statistics_checked=0
    max_numeric_delta=0.0
    for panel, datasets in read(OUT/'sources/two-city-v1/table-values.json').items():
        for ds,variants in datasets.items():
            for variant,metrics in variants.items():
                for metric,summary in metrics.items():
                    vals=[records[f'{ds}/{s}/{variant}']['metrics'][metric] for s in (135398,135399,135400)]
                    expected=(None,None) if any(x is None for x in vals) else (statistics.mean(vals),statistics.stdev(vals))
                    actual=(summary['mean'],summary['sd'])
                    for a,b in zip(actual,expected):
                        if a is not None and b is not None:
                            max_numeric_delta=max(max_numeric_delta,abs(a-b))
                        if not (a is None and b is None) and not (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)):
                            errors.append(f'statistics:{panel}/{ds}/{variant}/{metric}')
                    statistics_checked+=1
    save(OUT/'result-index.json',index)
    save(OUT/'local-audit.json',dict(state='passed' if not errors else 'failed',checked=dict(checked),
         summary_cells_recomputed=statistics_checked,max_numeric_delta=max_numeric_delta,
         statistics_tolerance=dict(relative=1e-12,absolute=1e-14),
         distinct_result_hashes=len(set(r['sha256'] for r in index)),errors=errors,
         note='Hashes and aggregate statistics verified in this delivery. Per-trajectory metric recomputation and merge equality are backed by the immutable server audits; no sampling/training rerun.'))
    print(json.dumps(dict(checked=dict(checked),statistics_checked=statistics_checked,errors=errors),ensure_ascii=False),flush=True)
    if errors: raise RuntimeError('Local audit failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['probe','deliver','audit'])
    args=parser.parse_args()
    {'probe':probe,'deliver':deliver,'audit':audit_local}[args.action]()
