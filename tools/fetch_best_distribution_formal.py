"""Fetch six sealed formal results without overwrite; verify hashes and CPU v3."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_io import atomic_json, safe_tag, sha256_file
from tools.fetch_category_geometry import safe_extract
from tools.fetch_distribution_study import PACK as LEGACY_PACK, SSH
from tools.run_best_distribution_formal import RUN_ID, verify_delivery

PACK = LEGACY_PACK.replace("files=sorted(audit['files'])", """
assert status['state']=='complete' and audit['results']==6 and audit['trajectory_records']==21066
for worker in root.glob('worker-formal-*.json'):
 if worker.name.endswith('.done.json'):continue
 item=json.loads(worker.read_text());assert item['state']=='complete'
 cmd=pathlib.Path('/proc')/str(item['pid'])/'cmdline'
 assert not cmd.exists() or root.name.encode() not in cmd.read_bytes(),'formal worker still running'
files=sorted(audit['files'])""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run = safe_tag(args.run_id)
    out = ROOT / 'experiment_runs' / run
    if args.verify_only:
        print(json.dumps(verify_delivery(out, recompute=True))); return
    if out.exists():
        raise RuntimeError('Local output exists; preserve it and use --verify-only')
    remote = f'/root/experiments/pcdg/{run}/experiment_runs/{run}'
    proc = subprocess.run(SSH + ['/root/anaconda3/envs/pcdg-exp/bin/python', '-B', '-'],
        input=PACK.replace('ROOT_VALUE', repr(remote)), text=True, encoding='utf-8', capture_output=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    server = json.loads(proc.stdout.strip().splitlines()[-1])
    (ROOT / 'tmp').mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='best-formal-fetch-', dir=ROOT / 'tmp'))
    archive = stage / 'delivery.tar.gz'
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                    'pcdg:' + server['archive'], str(archive)], check=True)
    if sha256_file(archive) != server['sha256']:
        raise RuntimeError('Transfer hash mismatch; staging retained')
    out.mkdir(parents=True, exist_ok=False)
    safe_extract(archive, out)
    receipt = verify_delivery(out, recompute=True)
    receipt.update(archive_sha256=server['sha256'], archive_path=str(archive), transfer_exited=True)
    atomic_json(out / 'local-delivery-audit.json', receipt)
    print(json.dumps(dict(receipt, destination=str(out))))


if __name__ == '__main__':
    main()
