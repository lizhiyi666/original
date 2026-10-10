from dataclasses import asdict
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiment_io import sha256_file
from tests.test_geometry_study import metric
from tools.geometry_inheritance import InheritedGeometry, SNAPSHOT, checked_path, verify_snapshot
from tools.geometry_study_common import CITIES, SEEDS, configurations, config_label
from tools.run_geometry_study import GeometryStudy

ROOT=Path(__file__).resolve().parents[1]


class GeometryInheritanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.old=self.root/'old';self.current=self.root/'new'
        self.source=self.old/'experiment_runs'/'old'
        self.source.mkdir(parents=True);self.current.mkdir()
        for root in (self.old,self.current):
            (root/'tools').mkdir()
            for name in ('run_geometry_study.py','geometry_study_common.py'):
                shutil.copyfile(ROOT/'tools'/name,root/'tools'/name)
            (root/'scientific.py').write_text('VALUE = 1\n',encoding='utf-8')
        self.manifest=dict(revision='pcdg-geo-v1',other_jsd_relative_tolerance=.05,run_id='old',
            source_sha256={p.relative_to(self.old).as_posix():sha256_file(p) for p in self.old.rglob('*.py')},
            splits={city:dict(screen=list(range(512)),confirm=list(range(512,1024))) for city in CITIES})
        self.save('manifest.json',self.manifest);self.manifest_sha=sha256_file(self.source/'manifest.json')
        self.save('status.json',dict(state='stopped',phase='stopped-at-quality-gate',pid=0))
        cases={city+'/'+case:dict(state='complete',ratio=1.04) for city in CITIES for case in ('ordinary','long')}
        self.save('performance-qualification.json',dict(selected_steps=200,ceiling=1.1,
            study_manifest_sha256=self.manifest_sha,speed={str(s):cases for s in (50,100,200)}))
        writer=object.__new__(GeometryStudy)
        self.screening=dict(results={config_label(c):dict(config=asdict(c),cities={}) for c in configurations(200)})
        for city in CITIES:
            for pool,seeds in (('screen',SEEDS[:1]),('confirm',SEEDS)):
                indices=self.manifest['splits'][city][pool]
                for seed in seeds:
                    prefix=f'calibration/{city}/{pool}/seed-{seed}'
                    self.save(prefix+'/baseline-metrics.json',dict(full=metric(),no_projection=metric()))
                    for offset in range(0,512,64):
                        identity=dict(study_manifest_sha256=self.manifest_sha,kind='base-cache',city=city,pool=pool,
                            seed=seed,indices=indices[offset:offset+64],global_start=offset+(512 if pool=='confirm' else 0))
                        writer.save_payload(self.source/prefix/f'batch-{offset}.pkl',identity,full=['unchanged'])
            for config in configurations(200):
                label=config_label(config)
                identity=dict(study_manifest_sha256=self.manifest_sha,kind='candidate',city=city,pool='screen',
                              seed=SEEDS[0],config=asdict(config))
                result=dict(state='complete',metrics=metric(),offline_seconds=1,config=asdict(config))
                writer.save_payload(self.source/f'candidates/{label}/{city}/screen/seed-{SEEDS[0]}.pkl',identity,result=result)
                self.screening['results'][label]['cities'][city]=[result]
        self.save('screening.json',self.screening)
        self.inventory=self.root/'source-inventory.json'
        self.seal()

    def tearDown(self):self.temporary.cleanup()

    def save(self,name,value):
        path=self.source/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')

    def seal(self):
        self.seal_record=dict(state='sealed',terminal_state='stopped',manifest_sha256=sha256_file(self.source/'manifest.json'),
            files={p.relative_to(self.source).as_posix():sha256_file(p) for p in self.source.rglob('*') if p.is_file()})
        self.inventory.write_text(json.dumps(self.seal_record),encoding='utf-8')

    def inherited(self):return InheritedGeometry(self.source,self.inventory,self.current)

    def identity(self):
        return dict(study_manifest_sha256='new',kind='base-cache',city=CITIES[0],pool='screen',seed=SEEDS[0],
                    indices=list(range(64)),global_start=0)

    def test_import_is_byte_preserving_original_identity_and_read_only(self):
        inherited=self.inherited()
        new_manifest=dict(self.manifest,run_id='new',source_sha256={},other_jsd_relative_tolerance=.1)
        desc=inherited.descriptor(new_manifest)
        output=self.current/'experiment_runs'/'new';inherited.materialize(output)
        self.assertEqual(desc['source_manifest_sha256'],self.manifest_sha)
        name=f'calibration/{CITIES[0]}/screen/seed-{SEEDS[0]}/batch-0.pkl'
        value=inherited.load_reusable(name,self.identity())
        self.assertEqual(value['identity']['study_manifest_sha256'],self.manifest_sha)
        self.assertEqual(value['full'],['unchanged'])
        self.assertEqual(sha256_file(self.source/name),sha256_file(output/SNAPSHOT/name))
        inherited.materialize(output)  # Idempotent recovery, no identity rewriting.
        inherited.verify_preserved()
        self.assertEqual(inherited.load_reusable('candidates/old/confirm/x.pkl',dict(kind='candidate',pool='confirm')),None)

    def test_changed_source_and_changed_scientific_code_are_rejected(self):
        (self.source/'status.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):self.inherited()

    def test_candidate_code_difference_is_rejected(self):
        (self.current/'scientific.py').write_text('VALUE = 2\n',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Scientific source differs'):self.inherited()

    def test_changed_sampler_method_is_rejected_even_in_coordinator(self):
        path=self.current/'tools'/'run_geometry_study.py'
        path.write_text(path.read_text(encoding='utf-8').replace('dd.projection_frequency=4','dd.projection_frequency=8'),encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Scientific study method differs'):self.inherited()

    def test_receipt_identity_is_checked_even_when_inventory_hash_is_valid(self):
        name=f'calibration/{CITIES[0]}/screen/seed-{SEEDS[0]}/batch-0.receipt.json'
        receipt=json.loads((self.source/name).read_text(encoding='utf-8'))
        receipt['identity']['seed']=42;self.save(name,receipt);self.seal()
        with self.assertRaisesRegex(RuntimeError,'receipt identity mismatch'):self.inherited()

    def test_protocol_difference_or_partial_screening_is_rejected(self):
        inherited=self.inherited()
        with self.assertRaisesRegex(RuntimeError,'configuration differs'):
            inherited.descriptor(dict(self.manifest,splits={},other_jsd_relative_tolerance=.1))
        self.screening['results'].pop(next(iter(self.screening['results'])))
        self.save('screening.json',self.screening);self.seal()
        with self.assertRaisesRegex(RuntimeError,'twelve screening'):self.inherited()

    def test_missing_source_cache_never_falls_back_to_regeneration(self):
        inherited=self.inherited()
        with self.assertRaisesRegex(RuntimeError,'Missing sealed'):
            inherited.load_reusable('calibration/missing.pkl',self.identity())
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
            inherited.load_reusable(f'calibration/{CITIES[0]}/screen/seed-{SEEDS[0]}/batch-0.pkl',
                                   dict(self.identity(),seed=42))

    def test_inherited_pool_bypasses_sampling_and_new_cache_writes(self):
        study=object.__new__(GeometryStudy);study.inherited=self.inherited()
        with patch.object(study,'read_pool',return_value=('cached','metrics')) as read_pool, \
             patch.object(study,'load_model') as load_model,patch.object(study,'spatial_sample') as sample:
            self.assertEqual(study.prepare_pool(CITIES[0],'confirm',SEEDS[1]),('cached','metrics'))
            read_pool.assert_called_once();load_model.assert_not_called();sample.assert_not_called()

    def test_import_conflict_preserves_existing_file(self):
        inherited=self.inherited();output=self.current/'experiment_runs'/'new'
        target=output/SNAPSHOT/'manifest.json';target.parent.mkdir(parents=True)
        target.write_text('preserved',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'different content'):inherited.materialize(output)
        self.assertEqual(target.read_text(encoding='utf-8'),'preserved')

    def test_path_escape_is_rejected(self):
        for name in ('../escape','/root/escape','a/../escape','a\\escape','C:/escape'):
            with self.assertRaises(RuntimeError):checked_path(self.source,name)


if __name__=='__main__':unittest.main()
