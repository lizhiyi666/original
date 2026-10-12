"""Deliver either a sealed calibration stop or the six-result formal study, then CPU-recompute."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_io import atomic_json, safe_tag, sha256_file
from tools.distribution_study_common import RUN_ID
from tools.fetch_category_geometry import safe_extract
from tools.run_distribution_study import verify

SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','pcdg']
PACK=r'''
import hashlib,json,pathlib,tarfile,tempfile,fcntl
root=pathlib.Path(ROOT_VALUE)
status=json.loads((root/'status.json').read_text());audit=json.loads((root/'audit.json').read_text())
assert status['state'] in ('stopped','complete') and audit['state']=='passed'
assert audit['terminal_state']==status['state']
assert audit['results']==(6 if status['state']=='complete' else 0)
pid=pathlib.Path('/proc')/str(status['pid'])/'cmdline'
assert not pid.exists() or root.name.encode() not in pid.read_bytes(), 'controller has not exited'
lock=(root/'pipeline.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
files=sorted(audit['files'])+['audit.json','status.json']
for name in files:
 rel=pathlib.PurePosixPath(name)
 assert not rel.is_absolute() and not any(p in ('','.','..') for p in name.split('/')) and ':' not in name and '\\' not in name
 path=root/name
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
 if name in audit['files']:assert sha(path)==audit['files'][name],name
folder=pathlib.Path(tempfile.mkdtemp(prefix='distribution-balance-fetch-'));archive=folder/'delivery.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in files:tar.add(root/name,arcname=name,recursive=False)
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),files=len(files))))
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',default=RUN_ID);ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();run=safe_tag(args.run_id);out=ROOT/'experiment_runs'/run
    if args.verify_only:
        print(json.dumps(verify(out,recompute=True)));return
    if out.exists():raise RuntimeError('Local output already exists; verify instead of overwriting')
    remote=f'/root/experiments/pcdg/{run}/experiment_runs/{run}'
    code=PACK.replace('ROOT_VALUE',repr(remote))
    proc=subprocess.run(SSH+['/root/anaconda3/envs/pcdg-exp/bin/python','-B','-'],input=code,
                        text=True,encoding='utf-8',capture_output=True,timeout=1800)
    if proc.returncode:raise RuntimeError(proc.stderr)
    server=json.loads(proc.stdout.strip().splitlines()[-1])
    (ROOT/'tmp').mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='distribution-balance-fetch-',dir=ROOT/'tmp'));archive=stage/'delivery.tar.gz'
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                    'pcdg:'+server['archive'],str(archive)],check=True)
    if sha256_file(archive)!=server['sha256']:raise RuntimeError('Transfer hash mismatch; staging retained')
    out.mkdir(parents=True);safe_extract(archive,out)
    receipt=verify(out,recompute=True)
    receipt.update(archive_sha256=server['sha256'],archive_path=str(archive),transfer_exited=True)
    atomic_json(out/'local-delivery-audit.json',receipt)
    print(json.dumps(dict(receipt,destination=str(out))))


if __name__=='__main__':main()
