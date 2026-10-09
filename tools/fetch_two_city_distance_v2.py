"""Download only sealed distance-v2 outputs, preserving every conflicting local file."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.integrate_experiment_results import digest, copy_no_overwrite
from experiment_io import safe_tag

OUT = ROOT / 'experiment_runs' / 'two-city-distance-v2'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', 'pcdg']


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or ':' in name or path.as_posix() != name):
        raise ValueError(f'Unsafe delivery path: {name}')
    return path


def check_completion(status, audit, registry):
    if (status.get('state') != 'server-complete' or audit.get('state') != 'passed'
            or audit.get('unique_results') != 66 or audit.get('trajectory_records') != 231726
            or len(registry) != 66):
        raise RuntimeError('Server has not completed and audited all 66 results')


REMOTE_BUILD = r'''
import hashlib, json, os, tarfile, tempfile
from pathlib import Path, PurePosixPath
root = Path(__REMOTE_RUN_ROOT__)
def read(name): return json.loads((root / name).read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()
status = read('status.json')
if status.get('state') != 'server-complete':
    raise RuntimeError('Outputs are not sealed; current phase: ' + status.get('phase', '?'))
audit, registry, seal = read('audit.json'), read('registry.json'), read('delivery-files.json')
if (audit.get('state') != 'passed' or audit.get('unique_results') != 66
        or audit.get('trajectory_records') != 231726 or len(registry) != 66 or seal.get('state') != 'sealed'):
    raise RuntimeError('Completion audit failed')
if sha(root / 'manifest.json') != seal['manifest_sha256']:
    raise RuntimeError('Manifest seal mismatch')
files = dict(seal['files'])
files['delivery-files.json'] = sha(root / 'delivery-files.json')
for name, expected in files.items():
    relative = PurePosixPath(name)
    path = root / name
    if (relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name
            or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file() or sha(path) != expected):
        raise RuntimeError('Unsafe or changed sealed file: ' + name)
manifest = read('manifest.json')
for name, expected in manifest['code_sha256'].items():
    if sha(root.parent.parent / name) != expected:
        raise RuntimeError('Deployed code changed: ' + name)
for record in registry.values():
    relative = Path(record['path']).relative_to(root).as_posix()
    if files.get(relative) != record['sha256']:
        raise RuntimeError('Unsealed registered result')
archive = root.parent.parent / ('delivery-' + files['delivery-files.json'][:20] + '.tar.gz')
if not archive.exists():
    fd, tmp = tempfile.mkstemp(prefix='delivery-', suffix='.part', dir=archive.parent)
    os.close(fd)
    with tarfile.open(tmp, 'w:gz') as tar:
        for name in sorted(files): tar.add(root / name, arcname=name, recursive=False)
    os.link(tmp, archive)
    os.unlink(tmp)
print(json.dumps(dict(archive=str(archive), archive_sha256=sha(archive),
                     files=len(files), bytes=archive.stat().st_size)))
'''


def verify(root):
    seal_path = root / 'delivery-files.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    if seal.get('state') != 'sealed':
        raise RuntimeError('Missing sealed inventory')
    for name, expected in seal['files'].items():
        path = root / str(safe_relative(name))
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or digest(path) != expected:
            raise RuntimeError(f'Local hash mismatch: {name}')
    if digest(root / 'manifest.json') != seal['manifest_sha256']:
        raise RuntimeError('Local manifest identity mismatch')
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    check_completion(read('status.json'), read('audit.json'), read('registry.json'))
    return dict(state='passed', manifest_sha256=seal['manifest_sha256'],
                inventory_sha256=digest(seal_path), verified_files=len(seal['files']) + 1,
                unique_results=66, trajectory_records=231726,
                original_server_metadata_preserved=True)


def extract_verified(archive, stage):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError('Duplicate archive members')
        for member in members:
            safe_relative(member.name)
            if not member.isfile():
                raise RuntimeError('Only regular output files may be delivered')
        seal = json.load(tar.extractfile('delivery-files.json'))
        if set(names) != set(seal['files']) | {'delivery-files.json'}:
            raise RuntimeError('Archive does not match the exact sealed inventory')
        for member in members:
            path = stage / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open('xb') as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)
    return verify(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--run-id', default='two-city-distance-v2')
    args = parser.parse_args()
    run_id = safe_tag(args.run_id)
    output = ROOT / 'experiment_runs' / run_id
    server_code = '/root/experiments/pcdg/' + run_id
    remote_code = REMOTE_BUILD.replace('__REMOTE_RUN_ROOT__', repr(server_code + '/experiment_runs/' + run_id))
    if args.verify_only:
        print(json.dumps(verify(output)))
        return
    proc = subprocess.run(SSH + ['/root/anaconda3/envs/pcdg-exp/bin/python', '-B', '-'],
                          input=remote_code, text=True, encoding='utf-8', capture_output=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    remote = json.loads(proc.stdout)
    if not remote['archive'].startswith(server_code + '/delivery-'):
        raise RuntimeError('Unexpected archive location')
    temporary_root = ROOT / 'tmp'
    temporary_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='distance-v2-delivery-', dir=temporary_root))
    archive = stage / 'delivery.tar.gz'
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                    'pcdg:' + remote['archive'], str(archive)], check=True)
    if digest(archive) != remote['archive_sha256']:
        raise RuntimeError('Transfer checksum mismatch; staging retained')
    contents = stage / 'contents'
    receipt = extract_verified(archive, contents)
    seal = json.loads((contents / 'delivery-files.json').read_text(encoding='utf-8'))
    files = dict(seal['files'], **{'delivery-files.json': digest(contents / 'delivery-files.json')})
    # Detect all collisions before publishing any result into the final directory.
    for name, expected in files.items():
        target = output / str(safe_relative(name))
        if not target.resolve().is_relative_to(output.resolve()) or target.is_symlink():
            raise RuntimeError(f'Unsafe existing destination: {target}')
        if target.exists() and digest(target) != expected:
            raise RuntimeError(f'Conflicting local file preserved: {target}')
    for name, expected in files.items():
        copy_no_overwrite(contents / name, output / name, expected)
    if verify(output) != receipt:
        raise RuntimeError('Final delivery differs from verified staging')
    proof = stage / 'local-delivery-audit.json'
    proof.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    copy_no_overwrite(proof, output / proof.name, digest(proof))
    print(json.dumps(dict(receipt, destination=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
