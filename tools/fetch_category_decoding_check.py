"""Hash-verified delivery of a completed 128-row category-decoding diagnostic."""
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
from tools.run_category_decoding_check import verify_delivery


REMOTE=r'''
import hashlib,json,os,pathlib,sys,tarfile,tempfile
root=pathlib.Path(__ROOT__)
sys.path.insert(0,str(root))
from tools.run_category_decoding_check import verify_delivery
out=root/'experiment_runs'/root.name
verify_delivery(out)
read=lambda p:json.loads(p.read_text())
status=read(out/'status.json')
pid=pathlib.Path('/proc')/str(status['pid'])
try:
 if (pid/'cwd').resolve()==root and b'run_category_decoding_check.py' in (pid/'cmdline').read_bytes():
  raise RuntimeError('Controller is still running')
except FileNotFoundError:pass
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=read(out/'manifest.json')
for name,expected in manifest['source_sha256'].items():
 if sha(root/name)!=expected:raise RuntimeError('Frozen code changed: '+name)
for name,expected in manifest['source_inputs'].items():
 if sha(pathlib.Path(name))!=expected:raise RuntimeError('Read-only input changed: '+name)
audit=read(out/'audit.json');audit_hash=sha(out/'audit.json')
archive=root/('category-delivery-'+audit_hash[:20]+'.tar.gz')
if not archive.exists():
 fd,tmp=tempfile.mkstemp(dir=root,suffix='.part');os.close(fd)
 with tarfile.open(tmp,'w:gz') as tar:
  for name in sorted(set(audit['files'])|{'audit.json','status.json'}):
   tar.add(out/name,arcname=name,recursive=False)
 os.link(tmp,archive);os.unlink(tmp)
print(json.dumps(dict(path=str(archive),sha256=sha(archive),audit_sha256=audit_hash,status_sha256=sha(out/'status.json'))))
'''


def verify_expected(folder,record):
    folder=Path(folder)
    if sha256_file(folder/'audit.json')!=record['audit_sha256'] or sha256_file(folder/'status.json')!=record['status_sha256']:
        raise RuntimeError('Server/local audit or status fingerprint differs')
    return verify_delivery(folder)


def extract(archive,destination,record):
    destination=Path(destination)
    if sha256_file(archive)!=record['sha256']:raise RuntimeError('Archive transfer hash mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise RuntimeError('Duplicate archive members')
        for member in members:
            if not member.isfile():raise RuntimeError('Only regular files may be delivered')
            checked_path(destination,member.name)
        audit=json.load(tar.extractfile('audit.json'))
        if set(names)!=set(audit['files'])|{'audit.json','status.json'}:
            raise RuntimeError('Archive contents differ from the audit')
        for member in members:
            target=checked_path(destination,member.name);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    return verify_expected(destination,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='ny-category-sampled-v2-20261010')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();tag=safe_tag(args.run_id);out=ROOT/'experiment_runs'/tag
    if args.verify_only:
        record=json.loads((out/'local-delivery-audit.json').read_text(encoding='utf-8'))
        summary=verify_expected(out,record['server'])
        if summary!=record['verification']:raise RuntimeError('Recorded local verification differs')
        print(json.dumps(summary));return
    remote='/root/experiments/pcdg/'+tag
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','pcdg',
        '/root/anaconda3/envs/pcdg-exp/bin/python','-B','-'],input=REMOTE.replace('__ROOT__',repr(remote)),
        text=True,encoding='utf-8',capture_output=True,timeout=1800)
    if result.returncode:raise RuntimeError(result.stderr)
    record=json.loads(result.stdout)
    if not record['path'].startswith(remote+'/category-delivery-'):raise RuntimeError('Unexpected remote archive')
    temporary=ROOT/'tmp';temporary.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='category-delivery-',dir=temporary));archive=stage/'delivery.tar.gz'
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                    'pcdg:'+record['path'],str(archive)],check=True)
    contents=stage/'contents';summary=extract(archive,contents,record)
    audit=json.loads((contents/'audit.json').read_text(encoding='utf-8'))
    files=dict(audit['files'],**{'audit.json':record['audit_sha256'],'status.json':record['status_sha256']})
    for name,expected in files.items():
        target=checked_path(out,name)
        if target.exists() and sha256_file(target)!=expected:raise RuntimeError('Conflicting local artifact retained: '+str(target))
    for name,expected in files.items():copy_no_overwrite(contents/name,checked_path(out,name),expected)
    if verify_expected(out,record)!=summary:raise RuntimeError('Final verification differs')
    proof=stage/'local-delivery-audit.json'
    proof.write_text(json.dumps(dict(server=record,verification=summary),indent=2)+'\n',encoding='utf-8')
    copy_no_overwrite(proof,out/proof.name,sha256_file(proof))
    print(json.dumps(dict(summary,destination=str(out))))


if __name__=='__main__':main()
