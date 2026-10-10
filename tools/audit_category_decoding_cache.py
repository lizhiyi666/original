"""Snapshot a failed completed sampler without edits, then audit against its original cache."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_io import safe_tag,sha256_file
from tools.geometry_inheritance import checked_path
from tools.integrate_experiment_results import copy_no_overwrite
from tools.category_cache_audit import REVISION,analyze_snapshot,report,verify_audit,verify_snapshot

SOURCE_ID='ny-category-sampled-v2-20261010'
SOURCE_MANIFEST='d9b6aa2b0468130801730631de873aed8c338d3615940caa5248d26a90f9e45b'
SOURCE_ON='39d0ad1d474468348ee6dc45b4d09a2fb0480eed8f6e6c05854308b702af3e78'

REMOTE=r'''
import hashlib,json,pathlib,shutil,sys,tarfile,tempfile
root=pathlib.Path('/root/experiments/pcdg/ny-category-sampled-v2-20261010')
out=root/'experiment_runs'/root.name
sys.path.insert(0,str(root))
import torch
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def require(condition,message):
 if not condition:raise RuntimeError(message)
require(sha(out/'manifest.json')==__MANIFEST__,'Unexpected original manifest')
require(sha(out/'on/payload.pkl')==__ON__,'Unexpected original on payload')
state=read(out/'status.json')
require(state['state']=='failed' and state['error']=='Generated length or record fields changed','Different original failure')
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  require(not ((p/'cwd').resolve()==root and b'run_category_decoding_check.py' in (p/'cmdline').read_bytes()),'Source sampler is running')
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
m=read(out/'manifest.json');parent=pathlib.Path(m['source_run']);parent_seal=read(parent/'delivery-files.json')
def verify_originals():
 for name,expected in m['source_sha256'].items():require(sha(root/name)==expected,'Source code changed: '+name)
 for name,expected in m['source_inputs'].items():require(sha(name)==expected,'Source input changed: '+name)
 for name,expected in parent_seal['files'].items():require(sha(parent/name)==expected,'Parent file changed: '+name)
verify_originals()
paths={p.relative_to(out).as_posix():p for p in out.rglob('*') if p.is_file() and p.name!='pipeline.lock'}
require(all(not p.is_symlink() for p in paths.values()),'Symlink in source')
before={name:sha(p) for name,p in paths.items()}
for payload,receipt in (('off.pkl','off.receipt.json'),('on/payload.pkl','on/receipt.json'),('derived-time-cache.pkl','derived-time-cache.receipt.json')):
 require(read(out/receipt)==dict(manifest_sha256=sha(out/'manifest.json'),sha256=sha(out/payload)),'Invalid source receipt: '+payload)
off=torch.load(out/'off.pkl',map_location='cpu',weights_only=False)
on=torch.load(out/'on/payload.pkl',map_location='cpu',weights_only=False)
derived=torch.load(out/'derived-time-cache.pkl',map_location='cpu',weights_only=False)
require(on['result']['state']=='complete' and on['result']['samples']==128,'Sampler payload incomplete')
require(on['job']['manifest_sha256']==sha(out/'manifest.json') and on['job']['cache_sha256']==sha(out/'derived-time-cache.pkl'),'On payload does not bind this manifest/cache')
origin=derived['derivation'];cache_path=pathlib.Path(origin['source_cache'])
require(m['source_inputs'].get(str(cache_path))==origin['source_sha256'],'Original cache is not a declared input')
require(sha(cache_path)==origin['source_sha256'],'Original time cache changed')
original_cache=torch.load(cache_path,map_location='cpu',weights_only=False)
shard_path=pathlib.Path(off['source']['path'])
require(m['source_inputs'].get(str(shard_path))==off['source']['sha256'],'Original Full shard is not a declared input')
require(sha(shard_path)==off['source']['sha256'],'Original Full shard changed')
shard=torch.load(shard_path,map_location='cpu',weights_only=False)
indices=m['indices'];require(indices==list(range(128)),'Unexpected selection');selected=set(indices)
items=[item for item in original_cache['batches'] if selected.intersection(item['indices'])]
require([i for item in items for i in item['indices']]==indices,'Not complete original time batches')
records=[];traces=[];offset=0
for trace in shard['batch_traces']:
 n=len(trace['indices']);values=shard['sequences'][offset:offset+n];offset+=n
 if selected.intersection(trace['indices']):records.extend(values);traces.append(trace)
require([i for t in traces for i in t['indices']]==indices and len(records)==128,'Original Full selection differs')
profile=on['job']['city'];data_path=pathlib.Path(profile['data_root'])/profile['dataset']/(profile['dataset']+'_test.pkl')
require(m['source_inputs'].get(str(data_path))==profile['data_sha256']['test'],'Reference data is not a declared input')
require(sha(data_path)==profile['data_sha256']['test'],'Reference data changed')
raw=torch.load(data_path,map_location='cpu',weights_only=False)
temp=pathlib.Path(tempfile.mkdtemp(prefix='category-cache-audit-'));snapshot=temp/'snapshot'
for name,path in paths.items():
 target=snapshot/'source/run'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
 require(sha(target)==before[name],'Snapshot copy differs: '+name)
log=root.parent/(root.name+'-controller-01.log')
log_hash=sha(log);shutil.copyfile(log,snapshot/'source/controller.log')
require(sha(snapshot/'source/controller.log')==log_hash,'Log copy differs')
bundle=dict(indices=indices,original_time_items=items,original_off_records=records,original_off_traces=traces,
 references=[raw['sequences'][i] for i in indices],poi_category=raw['poi_category'],poi_gps=raw['poi_gps'],
 reference_file_sha256=profile['data_sha256']['test'],original_cache_sha256=origin['source_sha256'],original_shard_sha256=off['source']['sha256'])
torch.save(bundle,snapshot/'source/original-inputs.pkl')
evaluation_files=[n for n in m['source_sha256'] if n.startswith('evaluations/') and n.endswith('.py')]
evaluation_files+=['tools/ablation_common.py','tools/perfcal_common.py','tools/baseline_common.py','experiment_io.py']
code={n:hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in evaluation_files}
verify_originals()
require({p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='pipeline.lock'}==before,'Source outputs changed during export')
provenance=dict(source_run=str(out),source_state=state,source_files=before,source_manifest_sha256=sha(out/'manifest.json'),
 on_payload_sha256=sha(out/'on/payload.pkl'),parent_files_verified=len(parent_seal['files']),
 source_inputs=m['source_inputs'],source_code_sha256=m['source_sha256'],evaluation_code_sha256=code,
 source_unchanged_after_export=True,controller_log_sha256=log_hash,no_sampling=True,no_training=True)
(snapshot/'source/provenance.json').write_text(json.dumps(provenance,sort_keys=True,indent=2)+'\n')
files={p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*') if p.is_file()}
seal=dict(state='sealed',source_state='failed',source_manifest_sha256=sha(out/'manifest.json'),files=files)
(snapshot/'snapshot-inventory.json').write_text(json.dumps(seal,sort_keys=True,indent=2)+'\n')
archive=temp/'snapshot.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in sorted(set(files)|{'snapshot-inventory.json'}):tar.add(snapshot/name,arcname=name,recursive=False)
print(json.dumps(dict(path=str(archive),sha256=sha(archive),inventory_sha256=sha(snapshot/'snapshot-inventory.json'),
 source_manifest_sha256=seal['source_manifest_sha256'],source_state='failed',files=len(files))))
'''


def extract_snapshot(archive,destination,server):
    destination=Path(destination)
    if sha256_file(archive)!=server['sha256']:raise RuntimeError('Snapshot transfer hash differs')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise RuntimeError('Duplicate snapshot member')
        for member in members:
            if not member.isfile():raise RuntimeError('Snapshot must contain regular files only')
            checked_path(destination,member.name)
        seal=json.load(tar.extractfile('snapshot-inventory.json'))
        if set(names)!=set(seal['files'])|{'snapshot-inventory.json'}:raise RuntimeError('Snapshot inventory differs')
        for member in members:
            target=checked_path(destination,member.name);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    if sha256_file(destination/'snapshot-inventory.json')!=server['inventory_sha256']:
        raise RuntimeError('Snapshot inventory fingerprint differs')
    verify_snapshot(destination)


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def finalize(stage,result,server):
    stage=Path(stage)
    save(stage/'alignment.json',result['alignment'])
    save(stage/'comparison.json',result['comparison'])
    save(stage/'per-condition.json',result['per_condition'])
    (stage/'report.md').write_text(report(result),encoding='utf-8')
    files={p.relative_to(stage).as_posix():sha256_file(p) for p in stage.rglob('*') if p.is_file()}
    save(stage/'audit.json',dict(state='passed',revision=REVISION,no_sampling=True,no_training=True,
        source_state='failed',source_manifest_sha256=result['source_manifest_sha256'],
        source_failure_preserved=True,old_export_length_invariant_passed=result['alignment']['restored_events']==0,
        restored_events=result['alignment']['restored_events'],traces=result['traces'],files=files,
        scientific_quality_passed=None,scope=result['scope'],server_snapshot=server,
        audit_code_sha256={name:sha256_file(ROOT/name) for name in ('tools/category_cache_audit.py','tools/audit_category_decoding_cache.py')}))
    save(stage/'status.json',dict(state='audit-complete',source_state='failed',no_sampling=True,
                                note='Independent time-cache audit, not a successful rerun.'))
    return verify_audit(stage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-id',default='ny-category-cache-audit-v1-20261010')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();out=ROOT/'experiment_runs'/safe_tag(args.audit_id)
    if args.verify_only:
        if (sha256_file(out/'source/run/manifest.json')!=SOURCE_MANIFEST
                or sha256_file(out/'source/run/on/payload.pkl')!=SOURCE_ON):
            raise RuntimeError('Pinned original source differs')
        result=verify_audit(out);receipt=json.loads((out/'local-delivery-audit.json').read_text(encoding='utf-8'))
        if result!=receipt:raise RuntimeError('Local delivery receipt differs')
        print(json.dumps(result));return
    if out.exists():raise RuntimeError('Audit output already exists; use --verify-only, never overwrite')
    code=REMOTE.replace('__MANIFEST__',repr(SOURCE_MANIFEST)).replace('__ON__',repr(SOURCE_ON))
    response=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','pcdg',
        '/root/anaconda3/envs/pcdg-exp/bin/python','-B','-'],input=code,text=True,encoding='utf-8',
        capture_output=True,timeout=600)
    if response.returncode:raise RuntimeError(response.stderr)
    server=json.loads(response.stdout)
    if (not server['path'].startswith('/tmp/category-cache-audit-') or '..' in Path(server['path']).parts
            or not server['path'].endswith('/snapshot.tar.gz') or server['source_manifest_sha256']!=SOURCE_MANIFEST):
        raise RuntimeError('Unexpected source snapshot')
    temporary=ROOT/'tmp';temporary.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='category-cache-audit-',dir=temporary));archive=stage/'snapshot.tar.gz'
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','pcdg:'+server['path'],str(archive)],check=True)
    work=stage/'audit';extract_snapshot(archive,work,server)
    if sha256_file(work/'source/run/on/payload.pkl')!=SOURCE_ON:raise RuntimeError('Wrong source on payload')
    result=analyze_snapshot(work,ROOT);summary=finalize(work,result,server)
    for path in work.rglob('*'):
        if path.is_file():copy_no_overwrite(path,checked_path(out,path.relative_to(work).as_posix()),sha256_file(path))
    if verify_audit(out)!=summary:raise RuntimeError('Final local audit differs')
    save(out/'local-delivery-audit.json',summary)
    print(json.dumps(dict(summary,restored_events=result['alignment']['restored_events'],destination=str(out))))


if __name__=='__main__':main()
