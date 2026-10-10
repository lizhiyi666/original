import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from experiment_io import sha256_file
from tools.fetch_category_decoding_check import REMOTE,extract,verify_expected


class CategoryDeliveryTests(unittest.TestCase):
    def archive(self,root):
        out=root/'experiment_runs'/root.name;out.mkdir(parents=True)
        for name,value in {'manifest.json':{'source_sha256':{},'source_inputs':{}},
            'comparison.json':{'on':{'category_poi_mismatch_rate':0.0}},
            'status.json':{'state':'complete','pid':0}}.items():
            (out/name).write_text(json.dumps(value),encoding='utf-8')
        files={name:sha256_file(out/name) for name in ('manifest.json','comparison.json')}
        (out/'audit.json').write_text(json.dumps(dict(state='passed',new_samples=128,baseline_samples=128,
            originals_unchanged=True,paired_rng=True,files=files)),encoding='utf-8')
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):exec(REMOTE.replace('__ROOT__',repr(str(root))),{})
        return json.loads(captured.getvalue())

    def test_server_archive_roundtrip_is_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=self.archive(root)
            summary=extract(record['path'],root/'download',record)
            self.assertEqual(summary['state'],'passed')
            self.assertEqual(summary['new_samples'],128)
            self.assertEqual(summary,verify_expected(root/'download',record))

    def test_archive_transfer_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=self.archive(root)
            with self.assertRaisesRegex(RuntimeError,'transfer hash'):
                extract(record['path'],root/'download',dict(record,sha256='bad'))
            self.assertFalse((root/'download').exists())

    def test_symlink_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'unsafe.tar.gz'
            with tarfile.open(path,'w:gz') as tar:
                item=tarfile.TarInfo('link');item.type=tarfile.SYMTYPE;item.linkname='/etc/passwd';tar.addfile(item)
            with self.assertRaisesRegex(RuntimeError,'regular'):
                extract(path,root/'download',{'sha256':sha256_file(path)})
            self.assertFalse((root/'download').exists())


if __name__=='__main__':unittest.main()
