"""CPU-only, immutable v3 re-evaluation of already sealed distribution-study records."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch

from evaluations.grank_v3 import EVALUATION_VERSION, REVISION, evaluate
from experiment_io import atomic_json, sha256_file
from tools.category_cache_audit import metrics_equal
from tools.distribution_study_common import (
    CITIES, SEEDS, METRICS, LOWER, HIGHER, aggregate, aligned_indices, label, read, select)
from distribution_balance import BalanceConfig
from tools.geometry_inheritance import checked_path

CHANGED_KEYS = {'G-RANK', 'totalJSD', 'evaluation_version'}
NEW_CODE = ('evaluations/grank_v3.py', 'tools/reevaluate_distribution_v3.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def disjoint(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise RuntimeError('Source and output must be disjoint directories')
    return source, output


class Source:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.audit = read(checked_path(self.root, 'audit.json'))
        self.status = read(checked_path(self.root, 'status.json'))
        terminal = self.status.get('state')
        if (self.audit.get('state') != 'passed' or terminal not in ('stopped', 'complete')
                or self.audit.get('terminal_state') != terminal
                or self.audit.get('results') != (6 if terminal == 'complete' else 0)):
            raise RuntimeError('Source requires a sealed terminal audit')
        if not {'manifest.json', 'summary.json', 'registry.json'} <= set(self.audit['files']):
            raise RuntimeError('Missing audited source metadata')
        self.snapshot = self.check()
        self.manifest = read(self.root / 'manifest.json')
        self.summary = read(self.root / 'summary.json')
        if self.summary['terminal_state'] != terminal:
            raise RuntimeError('Source summary/status mismatch')
        self.inputs = {}
        self.items = self.inventory()

    def check(self):
        hashes = {}
        for name, expected in self.audit['files'].items():
            actual = sha256_file(checked_path(self.root, name))
            if actual != expected:
                raise RuntimeError('Source hash mismatch: ' + name)
            hashes[name] = actual
        for name in ('audit.json', 'status.json'):
            hashes[name] = sha256_file(checked_path(self.root, name))
        if hasattr(self, 'snapshot') and hashes != self.snapshot:
            raise RuntimeError('Source changed during re-evaluation')
        return hashes

    def inventory(self):
        items = []
        phases = [p for p in ('screen', 'confirm', 'formal') if p in self.summary]
        if not phases or (self.status['state'] == 'stopped' and 'formal' in phases):
            raise RuntimeError('Invalid source phase inventory')
        for phase in phases:
            value = self.summary[phase]
            seeds = SEEDS[:1] if phase == 'screen' else SEEDS
            if set(value['baselines']) != set(CITIES):
                raise RuntimeError('Source baseline cities missing')
            for city in CITIES:
                rows = value['baselines'][city]['rows']
                if sorted(r['seed'] for r in rows) != list(seeds):
                    raise RuntimeError('Missing or duplicate baseline seed')
                for row in rows:
                    items.append(dict(path=f'baselines/{phase}/{city}/seed-{row["seed"]}.pkl',
                                      phase=phase, city=city, seed=row['seed'], method=None, stored=row))
            for name, candidate in sorted(value['results'].items()):
                if label(BalanceConfig(**candidate['config'])) != name or set(candidate['cities']) != set(CITIES):
                    raise RuntimeError('Source candidate configuration/cities mismatch')
                for city in CITIES:
                    rows = candidate['cities'][city]
                    if sorted(r['seed'] for r in rows) != list(seeds):
                        raise RuntimeError('Missing or duplicate candidate seed')
                    for row in rows:
                        items.append(dict(path=f'results/{phase}/{city}/seed-{row["seed"]}/{name}.pkl',
                                          phase=phase, city=city, seed=row['seed'], method=name, stored=row))
        expected = {x['path'] for x in items}
        actual = {p for p in self.audit['files'] if p.endswith('.pkl') and p.startswith(('results/', 'baselines/'))}
        if len(expected) != len(items) or expected != actual:
            raise RuntimeError('Result inventory differs from sealed summary/audit')
        return sorted(items, key=lambda x: x['path'])

    def packet(self, item):
        name = item['path']
        payload = torch.load(checked_path(self.root, name), map_location='cpu', weights_only=False)
        receipt_name = str(Path(name).with_suffix('.receipt.json')).replace('\\', '/')
        if receipt_name not in self.audit['files']:
            raise RuntimeError('Missing audited result receipt')
        identity = payload['identity']
        if read(checked_path(self.root, receipt_name)) != dict(identity=identity, sha256=self.snapshot[name]):
            raise RuntimeError('Result receipt differs from payload')
        if (identity.get('study_manifest_sha256') != self.snapshot['manifest.json']
                or any(identity.get(k) != item[k] for k in ('phase', 'city', 'seed'))
                or identity.get('kind') != ('result' if item['method'] else 'baseline')
                or payload['result'] != item['stored'] or payload['city'] != item['city']):
            raise RuntimeError('Result identity/summary mismatch')
        city, phase = item['city'], item['phase']
        split = 'test' if phase == 'formal' else 'train'
        if payload['split'] != split:
            raise RuntimeError('Result split mismatch')
        expected = (list(range(self.manifest['profiles'][city]['test_count'])) if phase == 'formal'
                    else self.manifest['splits'][city][phase])
        aligned_indices(payload['indices'], expected)
        aligned_indices(identity['indices'], expected)
        key = (city, split)
        if key not in self.inputs:
            path = f'evaluation_inputs/{city}/{city}_{split}.pkl'
            if self.snapshot.get(path) != self.manifest['profiles'][city]['data_sha256'][split]:
                raise RuntimeError('Evaluation input not bound to original data fingerprint')
            self.inputs[key] = torch.load(checked_path(self.root, path), map_location='cpu', weights_only=False)
        raw = self.inputs[key]
        refs = [raw['sequences'][i] for i in expected]
        values = {'candidate': payload['sequences']} if item['method'] else payload['sequences_by_method']
        if not item['method'] and set(values) != {'full', 'no_projection'}:
            raise RuntimeError('Baseline method inventory mismatch')
        return raw, refs, expected, values


def code_fingerprint(source):
    result, normalized = {}, []
    old = source.manifest['source_sha256']
    for name in sorted(set(old) | set(NEW_CODE)):
        data = checked_path(ROOT, name).read_bytes()
        lf = data.replace(b'\r\n', b'\n')
        raw_sha, lf_sha = hashlib.sha256(data).hexdigest(), hashlib.sha256(lf).hexdigest()
        if name in old and raw_sha != old[name]:
            # Text-only transport line endings may differ; dataset hashes never use this.
            candidates = {lf_sha, hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest()}
            if old[name] not in candidates:
                raise RuntimeError('Legacy code changed beyond line endings: ' + name)
            normalized.append(name)
        result[name] = dict(sha256=raw_sha, lf_sha256=lf_sha)
    return result, normalized


def records_for(source, progress=None):
    records = []
    for item in source.items:
        raw, refs, indices, methods = source.packet(item)
        for method, generated in sorted(methods.items()):
            stored = item['stored']['metrics'] if item['method'] else item['stored'][method]
            if stored.get('evaluation_version') != 2:
                raise RuntimeError('Re-evaluation expects original v2 metrics')
            detail = {}
            metrics, _ = evaluate(refs, generated, raw['poi_category'], indices, diagnostics=detail)
            metrics_equal({k: v for k, v in metrics.items() if k not in CHANGED_KEYS},
                          {k: v for k, v in stored.items() if k not in CHANGED_KEYS})
            records.append(dict(source_path=item['path'], source_sha256=source.snapshot[item['path']],
                phase=item['phase'], city=item['city'], seed=item['seed'],
                candidate=item['method'], method=method, samples=len(indices),
                legacy_metrics=stored, metrics=metrics, grank_diagnostics=detail))
            if progress:
                progress(len(records), item['path'], method)
    return records


def summary_for(source, records):
    summary = copy.deepcopy(source.summary)
    for record in records:
        phase, city, seed = record['phase'], record['city'], record['seed']
        if record['candidate']:
            rows = summary[phase]['results'][record['candidate']]['cities'][city]
            row = next(r for r in rows if r['seed'] == seed)
            row['metrics'] = record['metrics']
            row['grank_v3_diagnostics'] = record['grank_diagnostics']
        else:
            rows = summary[phase]['baselines'][city]['rows']
            next(r for r in rows if r['seed'] == seed)[record['method']] = record['metrics']
    for phase in ('screen', 'confirm', 'formal'):
        if phase not in summary:
            continue
        value = summary[phase]
        value['original_selection'] = value['selection']
        value['selection'] = select(value['results'], value['baselines'], confirmation=phase != 'screen')
        for candidate in value['results'].values():
            candidate['aggregate'] = {c: aggregate([r['metrics'] for r in candidate['cities'][c]]) for c in CITIES}
            candidate['paired_deltas'] = {}
            for city in CITIES:
                paired = {r['seed']: r for r in value['baselines'][city]['rows']}
                candidate['paired_deltas'][city] = [dict(seed=r['seed'], deltas={
                    k: (r['metrics'][k] - paired[r['seed']]['full' if k in LOWER + HIGHER else 'no_projection'][k]
                        if r['metrics'].get(k) is not None and paired[r['seed']]['full' if k in LOWER + HIGHER else 'no_projection'].get(k) is not None else None)
                    for k in METRICS}) for r in candidate['cities'][city]]
    old_top = [r['label'] for r in source.summary['screen']['selection']['ranking'][:2]]
    new_top = [r['label'] for r in summary['screen']['selection']['ranking'][:2]]
    summary.update(evaluation_version=EVALUATION_VERSION, evaluation_revision=REVISION,
                   interpretation='Retrospective re-evaluation of archived candidates only; no new validation or sampling.',
                   original_terminal_state=source.status['state'], original_quality_goals_met=source.audit['quality_goals_met'],
                   screening_shortlist_changed=old_top != new_top,
                   original_screening_shortlist=old_top, revised_screening_shortlist=new_top,
                   archived_confirm_candidates_meet_v3_gates=bool(summary.get('confirm', {}).get('selection', {}).get('ranking')))
    return summary


def report_text(summary):
    lines = ['# G-RANK v3：已有结果的独立复算', '',
             '仅复算封存轨迹；旧 v2、原始候选选择和实验终态不改写。不训练、不采样、不补跑确认、不启动正式测试。',
             '这是训练侧工程校准的回顾性复核，不是新增独立验证；均值通过不代表统计显著。', '',
             f"原终态：{summary['original_terminal_state']}；原质量达标：{summary['original_quality_goals_met']}。",
             f"v3 筛选前两名是否变化：{summary['screening_shortlist_changed']}。",
             f"已归档确认候选是否通过 v3 全门槛：{summary['archived_confirm_candidates_meet_v3_gates']}。", '',
             '所有基线与候选统一采用 v3；容差仍为 1e-12。表格为三种子均值 ± 样本标准差（筛选仅一个种子）。', '']
    for phase in ('screen', 'confirm', 'formal'):
        if phase not in summary:
            continue
        value = summary[phase]
        lines += [f'## {phase}', '', '| 配置 | 城市 | Distance | Radius | CategoryTransition | DailyLoc | Category | G-RANK | 失败项 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---|']
        for name, candidate in value['results'].items():
            for city in CITIES:
                cells = []
                for key in ('Distance', 'Radius', 'CategoryTransition', 'DailyLoc', 'Category', 'G-RANK'):
                    v = candidate['aggregate'][city][key]
                    cells.append('undefined' if v['mean'] is None else f"{v['mean']:.6f}" +
                                 (f" ± {v['sample_sd']:.6f}" if v['sample_sd'] is not None else ''))
                failures = [e.split(':', 1)[1] for e in value['selection']['failures'][name] if e.startswith(city + ':')]
                lines.append('| ' + name + ' | ' + city + ' | ' + ' | '.join(cells) + ' | ' + (', '.join(failures) or '无') + ' |')
        lines.append('')
    lines += ['旧/新逐记录指标和 top-1000 计数指纹：records.json。所有基线、约束、标准差与逐种子差值：summary.json。',
              '筛选排序变化只做审计记录；没有为新入选候选补跑确认。原 v2 的跨平台交付失败记录保持原样。', '']
    return '\n'.join(lines)


def run(source_path, output_path, evaluation_commit):
    source_path, output = disjoint(source_path, output_path)
    if output.exists():
        raise RuntimeError('Output exists; no overwrite or automatic resume')
    if not re.fullmatch(r'[0-9a-f]{40}', evaluation_commit):
        raise ValueError('Require verified GitHub evaluation commit SHA')
    source = Source(source_path)
    code, normalized = code_fingerprint(source)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(evaluation_version=EVALUATION_VERSION, revision=REVISION,
        evaluation_commit=evaluation_commit, original_experiment_commit=source.manifest['source_commit'],
        source_path=str(source_path), source_files=source.snapshot, source_snapshot_sha256=digest(source.snapshot),
        code_files=code, legacy_line_endings_only=normalized, top=1000, tolerance=1e-12,
        rule='count descending, integer POI ID ascending; union aligned by POI ID',
        no_training=True, no_sampling=True, no_new_confirmation=True, no_new_formal_results=True)
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'environment.json', dict(python=sys.version, numpy=np.__version__,
                torch=torch.__version__, platform=platform.platform(), device='cpu'))
    try:
        def progress(count, path, method):
            atomic_json(output / 'status.json', dict(state='running', pid=os.getpid(), metrics_recomputed=count,
                        last_source_path=path, last_method=method))
            print(json.dumps(dict(metrics_recomputed=count, source=path, method=method)), flush=True)
        records = records_for(source, progress)
        summary = summary_for(source, records)
        source.check()
        if code_fingerprint(source)[0] != code:
            raise RuntimeError('Evaluation code changed during run')
        atomic_json(output / 'records.json', records)
        atomic_json(output / 'summary.json', summary)
        (output / 'report.md').write_text(report_text(summary), encoding='utf-8', newline='\n')
        files = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'status.json'}
        atomic_json(output / 'audit.json', dict(state='passed', evaluation_version=EVALUATION_VERSION,
            files=files, source_unchanged=True, legacy_non_grank_metrics_equal=True, cpu_recomputed=True,
            metrics_recomputed=len(records), input_payloads=len(source.items), new_formal_results=0,
            source_snapshot_sha256=manifest['source_snapshot_sha256'], cross_platform_verified=False))
        atomic_json(output / 'status.json', dict(state='complete', metrics_recomputed=len(records), new_formal_results=0))
        return verify(output)
    except BaseException as exc:
        atomic_json(output / 'status.json', dict(state='failed', error_type=type(exc).__name__, error=str(exc)))
        raise


def verify(output_path, source_path=None, recompute=False):
    output = Path(output_path).resolve()
    audit = read(checked_path(output, 'audit.json'))
    if audit['state'] != 'passed' or read(checked_path(output, 'status.json'))['state'] != 'complete' or audit['evaluation_version'] != 3:
        raise RuntimeError('Invalid v3 output audit/status')
    for name, expected in audit['files'].items():
        if sha256_file(checked_path(output, name)) != expected:
            raise RuntimeError('v3 output hash mismatch: ' + name)
    records = read(output / 'records.json')
    if len(records) != audit['metrics_recomputed']:
        raise RuntimeError('v3 record count mismatch')
    count = 0
    if source_path is not None:
        source = Source(source_path)
        manifest = read(output / 'manifest.json')
        if source.snapshot != manifest['source_files']:
            raise RuntimeError('v3 source snapshot differs')
        if recompute:
            actual = records_for(source)
            compare_records(records, actual)
            source.check()
            count = len(actual)
    elif recompute:
        raise ValueError('Recompute requires the original source directory')
    return dict(state='passed', verified_files=len(audit['files']) + 2, records=len(records),
                metrics_recomputed=count, source_checked=source_path is not None)


def compare_records(left, right):
    if len(left) != len(right):
        raise RuntimeError('Cross-platform record counts differ')
    maximum = 0.0
    for a, b in zip(left, right):
        for key in ('source_path', 'source_sha256', 'phase', 'city', 'seed', 'candidate', 'method',
                    'samples', 'legacy_metrics', 'grank_diagnostics'):
            if a[key] != b[key]:
                raise RuntimeError('Cross-platform identity/selection differs: ' + key)
        metrics_equal(a['metrics'], b['metrics'])
        maximum = max([maximum] + [abs(v - b['metrics'][k]) for k, v in a['metrics'].items() if v is not None])
    return maximum


def compare(left, right):
    verify(left)
    verify(right)
    lm, rm = (read(Path(p) / 'manifest.json') for p in (left, right))
    for key in ('evaluation_version', 'revision', 'evaluation_commit', 'source_snapshot_sha256',
                'source_files', 'top', 'tolerance', 'rule'):
        if lm[key] != rm[key]:
            raise RuntimeError('Cross-platform manifest mismatch: ' + key)
    if {k: v['lf_sha256'] for k, v in lm['code_files'].items()} != {k: v['lf_sha256'] for k, v in rm['code_files'].items()}:
        raise RuntimeError('Cross-platform code fingerprint differs')
    le, re_ = (read(Path(p) / 'environment.json') for p in (left, right))
    if {le['platform'].split('-')[0], re_['platform'].split('-')[0]} != {'Windows', 'Linux'}:
        raise RuntimeError('Cross-platform audit requires Windows and Linux evidence')
    lr, rr = (read(Path(p) / 'records.json') for p in (left, right))
    maximum = compare_records(lr, rr)
    ls, rs = (read(Path(p) / 'summary.json') for p in (left, right))
    for phase in ('screen', 'confirm', 'formal'):
        if (phase in ls) != (phase in rs):
            raise RuntimeError('Cross-platform phases differ')
        if phase in ls:
            a, b = ls[phase]['selection'], rs[phase]['selection']
            if a['failures'] != b['failures'] or [r['label'] for r in a['ranking']] != [r['label'] for r in b['ranking']]:
                raise RuntimeError('Cross-platform gate or ranking differs')
    return dict(state='passed', evaluation_version=3, revision=REVISION, metrics_compared=len(lr),
        max_abs_metric_difference=maximum, tolerance=1e-12, top1000_fingerprints_equal=True,
        source_snapshot_sha256=lm['source_snapshot_sha256'], evaluation_commit=lm['evaluation_commit'],
        windows_linux_verified=True, original_v2_rewritten=False, new_formal_results=0,
        left_audit_sha256=sha256_file(Path(left) / 'audit.json'), right_audit_sha256=sha256_file(Path(right) / 'audit.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('run')
    create.add_argument('--source', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--evaluation-commit', required=True)
    check = commands.add_parser('verify')
    check.add_argument('--output', type=Path, required=True)
    check.add_argument('--source', type=Path)
    check.add_argument('--recompute', action='store_true')
    match = commands.add_parser('compare')
    match.add_argument('--left', type=Path, required=True)
    match.add_argument('--right', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run(args.source, args.output, args.evaluation_commit)
    elif args.command == 'verify':
        result = verify(args.output, args.source, args.recompute)
    else:
        result = compare(args.left, args.right)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
