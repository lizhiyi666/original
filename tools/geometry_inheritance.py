"""Read-only, sealed reuse of geometry-v1 evidence under a new selection policy.

Imported payloads keep their original identities. Only base caches and screening
candidates are reusable; old confirmation rankings never select new finalists.
"""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch

from experiment_io import sha256_file
from tools.geometry_study_common import CITIES, SEEDS, choose_steps, config_label, configurations
from tools.integrate_experiment_results import copy_no_overwrite


SNAPSHOT = 'inherited/geometry-v1'
POLICY = 'pcdg-geo-selection-tol10-v1'
# These files coordinate selection/delivery or a standalone worker not called by
# this study. Scientific routines in the two coordinator files are AST-pinned.
COORDINATION_FILES = {
    'tools/run_geometry_study.py', 'tools/geometry_study_common.py',
    'tools/ablation_worker.py', 'tools/fetch_geometry_study.py',
}
FROZEN_METHODS = (
    'load_model', 'finish_model', 'memory_guard', 'configure_sampler',
    'spatial_sample', 'speed_cases', 'measure_case', 'profile_tail',
    'refine_pool', 'source_batches', 'formal',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_path(root, name):
    relative = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or relative.is_absolute()
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise RuntimeError('Unsafe inherited path: '+name)
    path = Path(root).joinpath(*relative.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise RuntimeError('Inherited path escapes its root: '+name)
    if any(p.is_symlink() for p in (path, *path.parents) if p != Path(root).parent):
        raise RuntimeError('Symlink in inherited path: '+name)
    return path


def verify_files(root, seal):
    if seal.get('state') != 'sealed' or seal.get('terminal_state') != 'stopped':
        raise RuntimeError('Inheritance requires the sealed stopped geometry-v1 run')
    for name, expected in seal['files'].items():
        if sha256_file(checked_path(root, name)) != expected:
            raise RuntimeError('Inherited file hash mismatch: '+name)
    if sha256_file(Path(root)/'manifest.json') != seal['manifest_sha256']:
        raise RuntimeError('Inherited manifest hash mismatch')
    status = read(Path(root)/'status.json')
    if status.get('state') != 'stopped' or status.get('phase') != 'stopped-at-quality-gate':
        raise RuntimeError('Source is not stopped at the quality gate')


def verify_snapshot(root):
    root = Path(root)
    seal = read(root/'delivery-inventory.json')
    verify_files(root, seal)
    return len(seal['files'])


def ast_members(path, class_name=None):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = tree.body if class_name is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: ast.dump(n, include_attributes=False) for n in nodes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def verify_scientific_code(source_root, current_root, original):
    source_root, current_root = Path(source_root), Path(current_root)
    checked = {}
    for name, expected in original['source_sha256'].items():
        old = checked_path(source_root, name)
        if sha256_file(old) != expected:
            raise RuntimeError('Source deployment changed: '+name)
        if name.startswith('tests/') or name in COORDINATION_FILES:
            continue
        before = old.read_bytes().replace(b'\r\n', b'\n')
        after = checked_path(current_root, name).read_bytes().replace(b'\r\n', b'\n')
        if before != after:
            raise RuntimeError('Scientific source differs: '+name)
        checked[name] = hashlib.sha256(after).hexdigest()
    name = 'tools/run_geometry_study.py'
    old, new = (ast_members(p/name, 'GeometryStudy') for p in (source_root, current_root))
    for method in FROZEN_METHODS:
        if old[method] != new[method]:
            raise RuntimeError('Scientific study method differs: '+method)
    name = 'tools/geometry_study_common.py'
    old, new = (ast_members(p/name) for p in (source_root, current_root))
    for method in old.keys() - {'quality_failures', 'rank_candidates'}:
        if old[method] != new.get(method):
            raise RuntimeError('Scientific common function differs: '+method)
    return checked


class InheritedGeometry:
    def __init__(self, source_run, inventory, current_root):
        self.source = Path(source_run).resolve()
        self.inventory = Path(inventory).resolve()
        self.seal = read(self.inventory)
        verify_files(self.source, self.seal)
        self.original = read(self.source/'manifest.json')
        if (self.original.get('revision') != 'pcdg-geo-v1'
                or self.original.get('other_jsd_relative_tolerance') != .05
                or 'inheritance' in self.original):
            raise RuntimeError('Expected the original 5-percent geometry-v1 protocol')
        # Do not copy a changing process, including one whose status is stale.
        pid = Path('/proc')/str(read(self.source/'status.json').get('pid', 0))
        if pid.exists() and (pid/'cwd').resolve() == self.source.parent.parent:
            raise RuntimeError('Source controller has not exited')
        self.code = verify_scientific_code(self.source.parent.parent, current_root, self.original)
        self.root = self.source
        self.validate_required_evidence()

    def validate_required_evidence(self):
        performance = read(self.file('performance-qualification.json'))
        if (performance['study_manifest_sha256'] != self.seal['manifest_sha256']
                or performance['selected_steps'] != 200
                or choose_steps(performance['speed']) != 200
                or performance['ceiling'] != 1.10):
            raise RuntimeError('Invalid inherited 200-step qualification')
        screening = read(self.file('screening.json'))
        expected = {config_label(c): c for c in configurations(200)}
        if set(screening['results']) != set(expected):
            raise RuntimeError('Inheritance requires all twelve screening candidates')
        # Check every base cache identity, not merely the summary JSON.
        for city in CITIES:
            for pool, seeds in (('screen', SEEDS[:1]), ('confirm', SEEDS)):
                indices = self.original['splits'][city][pool]
                for seed in seeds:
                    for offset in range(0, 512, 64):
                        identity = dict(study_manifest_sha256=self.seal['manifest_sha256'],
                            kind='base-cache', city=city, pool=pool, seed=seed,
                            indices=indices[offset:offset+64], global_start=offset+(512 if pool=='confirm' else 0))
                        self.payload(f'calibration/{city}/{pool}/seed-{seed}/batch-{offset}.pkl', identity)
                    self.file(f'calibration/{city}/{pool}/seed-{seed}/baseline-metrics.json')
            for label, config in expected.items():
                from dataclasses import asdict
                identity = dict(study_manifest_sha256=self.seal['manifest_sha256'], kind='candidate',
                                city=city, pool='screen', seed=SEEDS[0], config=asdict(config))
                payload = self.payload(f'candidates/{label}/{city}/screen/seed-{SEEDS[0]}.pkl', identity)
                if screening['results'][label]['cities'][city] != [payload['result']]:
                    raise RuntimeError('Screening summary differs from sealed candidate: '+label)

    def descriptor(self, manifest):
        for name, expected in self.original.items():
            if name in ('run_id', 'source_sha256', 'other_jsd_relative_tolerance'):
                continue
            if manifest.get(name) != expected:
                raise RuntimeError('Inherited scientific configuration differs: '+name)
        if manifest['other_jsd_relative_tolerance'] != .10:
            raise RuntimeError('This inheritance revision is specifically the 10-percent protocol')
        return dict(source_run=str(self.source), source_manifest_sha256=self.seal['manifest_sha256'],
                    source_inventory_sha256=sha256_file(self.inventory), snapshot=SNAPSHOT,
                    normalized_scientific_sha256=self.code,
                    reuse='all base caches, twelve screening candidates, initial 200-step qualification; no confirmation reuse')

    def materialize(self, output):
        destination = Path(output)/SNAPSHOT
        if destination.resolve().is_relative_to(self.source):
            raise RuntimeError('Cannot import into source run')
        for name, expected in self.seal['files'].items():
            copy_no_overwrite(self.file(name), checked_path(destination, name), expected)
        copy_no_overwrite(self.inventory, destination/'delivery-inventory.json', sha256_file(self.inventory))
        verify_snapshot(destination)
        self.root = destination

    def file(self, name):
        if name not in self.seal['files']:
            raise RuntimeError('Missing sealed inherited evidence: '+name)
        path = checked_path(self.root, name)
        if sha256_file(path) != self.seal['files'][name]:
            raise RuntimeError('Inherited file hash mismatch: '+name)
        return path

    def payload(self, name, identity):
        original_identity = dict(identity, study_manifest_sha256=self.seal['manifest_sha256'])
        path = self.file(name)
        receipt = read(self.file(str(PurePosixPath(name).with_suffix('.receipt.json'))))
        if receipt != dict(identity=original_identity, sha256=self.seal['files'][name]):
            raise RuntimeError('Inherited receipt identity mismatch: '+name)
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value.get('identity') != original_identity:
            raise RuntimeError('Inherited payload identity mismatch: '+name)
        return value

    def load_reusable(self, name, identity):
        allowed = (identity.get('kind') == 'base-cache' and name.startswith('calibration/')) or (
            identity.get('kind') == 'candidate' and identity.get('pool') == 'screen' and name.startswith('candidates/'))
        return self.payload(name, identity) if allowed else None

    def verify_preserved(self):
        verify_files(self.source, self.seal)
        verify_snapshot(self.root)
