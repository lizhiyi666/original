"""Fetch a terminal PCDG-Geo study without changing its sealed source artifacts."""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_io import safe_tag
from tools.fetch_two_city_distance_v2 import safe_relative
from tools.integrate_experiment_results import copy_no_overwrite, digest


REMOTE=r'''
import hashlib,io,json,os,pathlib,tarfile,tempfile
root=pathlib.Path(__ROOT__)
out=root/'experiment_runs'/root.name
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
status=read(out/'status.json')
if status['state'] not in ('complete','stopped'):raise RuntimeError('Study is not terminal: '+status['state'])
pid=pathlib.Path('/proc')/str(status['pid'])
try:
 if (pid/'cwd').resolve()==root and b'run_geometry_study.py' in (pid/'cmdline').read_bytes():raise RuntimeError('Controller has not exited')
except FileNotFoundError:pass
manifest=read(out/'manifest.json')
for name,expected in manifest['source_sha256'].items():
 if sha(root/name)!=expected:raise RuntimeError('Deployed source changed: '+name)
if status['state']=='complete':
 audit=read(out/'audit.json')
 if audit['state']!='passed' or audit['new_results']!=18 or len(read(out/'registry.json'))!=18:raise RuntimeError('Incomplete formal result audit')
 for name,expected in audit['files'].items():
  if sha(out/name)!=expected:raise RuntimeError('Formal artifact changed: '+name)
elif status['phase'] not in ('stopped-at-speed-gate','stopped-at-quality-gate'):
 raise RuntimeError('Unknown stop condition')
files={}
for p in sorted(out.rglob('*')):
 if p.name=='pipeline.lock':continue
 if p.is_symlink():raise RuntimeError('Symlink in output')
 if p.is_file():files[p.relative_to(out).as_posix()]=sha(p)
seal=dict(state='sealed',terminal_state=status['state'],manifest_sha256=sha(out/'manifest.json'),files=files)
encoded=(json.dumps(seal,sort_keys=True,indent=2)+'\n').encode()
seal_hash=hashlib.sha256(encoded).hexdigest()
archive=root/('geometry-delivery-'+seal_hash[:20]+'.tar.gz')
if not archive.exists():
 fd,tmp=tempfile.mkstemp(dir=root,suffix='.part');os.close(fd)
 with tarfile.open(tmp,'w:gz') as tar:
  for name in sorted(files):tar.add(out/name,arcname=name,recursive=False)
  info=tarfile.TarInfo('delivery-inventory.json');info.size=len(encoded);tar.addfile(info,io.BytesIO(encoded))
 os.link(tmp,archive);os.unlink(tmp)
print(json.dumps(dict(path=str(archive),sha256=sha(archive),inventory_sha256=seal_hash)))
'''


def verify(root):
    root=Path(root)
    seal=json.loads((root/'delivery-inventory.json').read_text(encoding='utf-8'))
    if seal.get('state')!='sealed':raise RuntimeError('Missing delivery seal')
    for name,expected in seal['files'].items():
        relative=safe_relative(name);path=root/str(relative)
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or digest(path)!=expected:
            raise RuntimeError('Delivered file hash/path mismatch: '+name)
    if digest(root/'manifest.json')!=seal['manifest_sha256']:raise RuntimeError('Manifest hash mismatch')
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    if status['state']!=seal['terminal_state']:raise RuntimeError('Terminal state mismatch')
    count=0
    if status['state']=='complete':
        audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
        count=len(json.loads((root/'registry.json').read_text(encoding='utf-8')))
        if audit['state']!='passed' or audit['new_results']!=18 or count!=18:raise RuntimeError('Formal audit incomplete')
    elif status['state']!='stopped' or status['phase'] not in ('stopped-at-speed-gate','stopped-at-quality-gate'):
        raise RuntimeError('Unexpected terminal state')
    return dict(state='passed',terminal_state=status['state'],formal_results=count,
                quality_goals_met=status.get('quality_goals_met'),verified_files=len(seal['files'])+1,
                manifest_sha256=seal['manifest_sha256'],inventory_sha256=digest(root/'delivery-inventory.json'))


def extract(archive,stage):
    stage=Path(stage)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise RuntimeError('Duplicate archive member')
        for member in members:
            safe_relative(member.name)
            if not member.isfile():raise RuntimeError('Archive must contain only regular files')
        seal=json.load(tar.extractfile('delivery-inventory.json'))
        if set(names)!=set(seal['files'])|{'delivery-inventory.json'}:raise RuntimeError('Inventory differs from archive')
        for member in members:
            target=stage/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    return verify(stage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='pcdg-geo-v1-20261010')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();tag=safe_tag(args.run_id)
    output=ROOT/'experiment_runs'/tag
    if args.verify_only:
        print(json.dumps(verify(output)));return
    server_root='/root/experiments/pcdg/'+tag
    code=REMOTE.replace('__ROOT__',repr(server_root))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','pcdg',
        '/root/anaconda3/envs/pcdg-exp/bin/python','-B','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=1800)
    if result.returncode:raise RuntimeError(result.stderr)
    record=json.loads(result.stdout)
    if not record['path'].startswith(server_root+'/geometry-delivery-'):raise RuntimeError('Foreign delivery archive')
    temporary=ROOT/'tmp';temporary.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='geometry-delivery-',dir=temporary));archive=stage/'delivery.tar.gz'
    subprocess.run(['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','pcdg:'+record['path'],str(archive)],check=True)
    if digest(archive)!=record['sha256']:raise RuntimeError('Transfer hash mismatch; staging retained')
    contents=stage/'contents';receipt=extract(archive,contents)
    if receipt['inventory_sha256']!=record['inventory_sha256']:raise RuntimeError('Inventory transfer mismatch')
    seal=json.loads((contents/'delivery-inventory.json').read_text(encoding='utf-8'))
    files=dict(seal['files'],**{'delivery-inventory.json':receipt['inventory_sha256']})
    for name,expected in files.items():
        path=output/str(safe_relative(name))
        if not path.resolve().is_relative_to(output.resolve()) or path.is_symlink():raise RuntimeError('Unsafe destination')
        if path.exists() and digest(path)!=expected:raise RuntimeError('Conflicting local file preserved: '+str(path))
    for name,expected in files.items():copy_no_overwrite(contents/name,output/name,expected)
    if verify(output)!=receipt:raise RuntimeError('Final delivery verification differs')
    proof=stage/'local-delivery-audit.json';proof.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    copy_no_overwrite(proof,output/proof.name,digest(proof))
    print(json.dumps(dict(receipt,destination=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
