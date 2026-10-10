import hashlib
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.fetch_geometry_study import REMOTE,extract,verify


class GeometryDeliveryTests(unittest.TestCase):
    def test_remote_builder_creates_a_valid_local_verifiable_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            out=root/'experiment_runs'/root.name;out.mkdir(parents=True)
            (out/'manifest.json').write_text(json.dumps({'source_sha256':{}}),encoding='utf-8')
            (out/'status.json').write_text(json.dumps({'state':'stopped','phase':'stopped-at-speed-gate','pid':0}),encoding='utf-8')
            (out/'report.md').write_text('No formal run started.',encoding='utf-8')
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured):exec(REMOTE.replace('__ROOT__',repr(str(root))),{})
            receipt=json.loads(captured.getvalue())
            result=extract(receipt['path'],root/'delivered')
            self.assertEqual(result['inventory_sha256'],receipt['inventory_sha256'])
            self.assertEqual(result['terminal_state'],'stopped')

    def archive(self,root,complete=False):
        encode=lambda x:json.dumps(x).encode()
        content={'manifest.json':encode({'revision':'pcdg-geo-v1'}),
                 'status.json':encode({'state':'complete' if complete else 'stopped','phase':'stopped-at-speed-gate'}),
                 'report.md':b'All original artifacts preserved.'}
        if complete:
            content['registry.json']=encode({str(i):{} for i in range(18)})
            content['audit.json']=encode(dict(state='passed',new_results=18))
        hashes={k:hashlib.sha256(v).hexdigest() for k,v in content.items()}
        content['delivery-inventory.json']=encode(dict(state='sealed',terminal_state='complete' if complete else 'stopped',
                                                      manifest_sha256=hashes['manifest.json'],files=hashes))
        archive=root/'result.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for name,value in content.items():
                info=tarfile.TarInfo(name);info.size=len(value);tar.addfile(info,io.BytesIO(value))
        return archive

    def test_both_gate_stop_and_full_completion_are_verifiable(self):
        for complete in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);archive=self.archive(root,complete)
                result=extract(archive,root/'out')
                self.assertEqual(result['state'],'passed')
                self.assertEqual(result['formal_results'],18 if complete else 0)
                self.assertEqual(result,verify(root/'out'))

    def test_changed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);extract(self.archive(root),root/'out')
            (root/'out'/'report.md').write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'hash/path'):verify(root/'out')

    def test_symlink_archive_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'unsafe.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                member=tarfile.TarInfo('link');member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd';tar.addfile(member)
            with self.assertRaisesRegex(RuntimeError,'regular'):extract(archive,root/'out')
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
