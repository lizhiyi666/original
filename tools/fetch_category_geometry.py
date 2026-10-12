"""Terminal download + safe unpack + independent local recheck for the category-geometry run.

Pulls the finished server output as a tar of regular files only, verifies the transfer hash,
extracts through a hardened safe-path check, then re-verifies the sealed audit (6 results,
every file hash, quality flag). It never generates results, loads a model, or edits the
source. It does NOT carry the old geometry study's 18-result acceptance logic.
"""
import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_io import safe_tag, sha256_file
from tools.geometry_inheritance import checked_path

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', 'pcdg']
REMOTE = '/root/experiments/pcdg/{run}/experiment_runs/{run}'

REMOTE_PACK = r'''
import hashlib, json, os, pathlib, tarfile, tempfile
root = pathlib.Path({root!r})
status = json.loads((root/'status.json').read_text()); audit = json.loads((root/'audit.json').read_text())
assert status.get('state') == 'complete' and audit.get('state') == 'passed' and audit.get('results') == 6, 'not a complete 6-result pass'
files = sorted(audit['files']) + ['audit.json', 'status.json']
tmp = tempfile.mkdtemp(prefix='category-geo-fetch-'); archive = pathlib.Path(tmp)/'delivery.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for name in files:
        p = root/name
        assert p.is_file() and not p.is_symlink(), name
        tar.add(p, arcname=name, recursive=False)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
print(json.dumps(dict(archive=str(archive), sha256=sha(archive), files=len(files))))
'''


def verify(out):
    audit = json.loads((out / 'audit.json').read_text(encoding='utf-8'))
    if audit.get('state') != 'passed' or audit.get('results') != 6:
        raise RuntimeError('Audit is not a complete 6-result pass')
    for name, expected in audit['files'].items():
        target = checked_path(out, name)
        if sha256_file(target) != expected:
            raise RuntimeError('Local hash mismatch: ' + name)
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    return dict(state='passed', verified_files=len(audit['files']) + 2, results=6,
                quality_goals_met=audit.get('quality_goals_met'),
                summary_sha256=sha256_file(out / 'summary.json'),
                cities=sorted(summary.keys()))


def safe_extract(archive, out):
    """Extract regular files only, each through the hardened safe-path check; reject duplicates."""
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise RuntimeError('Duplicate archive member')
        for member in members:
            if not member.isfile():
                raise RuntimeError('Only regular files may be delivered')
            target = checked_path(out, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open('xb') as dst:
                dst.write(src.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='ny-ist-category-geo-fixed-v1-20261012')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run = safe_tag(args.run_id)
    out = ROOT / 'experiment_runs' / run
    if args.verify_only:
        print(json.dumps(verify(out)))
        return
    remote_root = REMOTE.format(run=run)
    code = REMOTE_PACK.format(root=remote_root)
    proc = subprocess.run(SSH + ['/root/anaconda3/envs/pcdg-exp/bin/python', '-B', '-'],
                          input=code, text=True, encoding='utf-8', capture_output=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    server = json.loads(proc.stdout.strip().splitlines()[-1])
    temporary = ROOT / 'tmp'
    temporary.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='category-geo-fetch-', dir=temporary))
    archive = stage / 'delivery.tar.gz'
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                    'pcdg:' + server['archive'], str(archive)], check=True)
    if sha256_file(archive) != server['sha256']:
        raise RuntimeError('Transfer checksum mismatch; staging retained')
    if out.exists():
        raise RuntimeError('Local output already exists; refusing to overwrite')
    out.mkdir(parents=True)
    safe_extract(archive, out)
    receipt = verify(out)
    (out / 'local-delivery-audit.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(receipt, destination=str(out))))


if __name__ == '__main__':
    main()
