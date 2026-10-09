import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.fetch_two_city_distance_v2 import (check_completion, extract_verified,
                                            safe_relative, verify)


class DistanceDeliveryTests(unittest.TestCase):
    def test_rejects_escaping_and_platform_specific_paths(self):
        for name in ('', '/etc/passwd', '../x', 'a/../../x', 'C:/x', 'a\\x', './x', 'a//x'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)
        self.assertEqual(str(safe_relative('NewYork_PO1_OOD/seed-135398/full/generated.pkl')),
                         'NewYork_PO1_OOD/seed-135398/full/generated.pkl')

    def test_incomplete_or_unaudited_run_is_never_delivered(self):
        status = dict(state='server-complete')
        audit = dict(state='passed', unique_results=66, trajectory_records=231726)
        records = {str(i): {} for i in range(66)}
        check_completion(status, audit, records)
        for s, a, r in ((dict(state='running'), audit, records),
                        (status, dict(audit, state='failed'), records),
                        (status, audit, {})):
            with self.assertRaises(RuntimeError):
                check_completion(s, a, r)

    def test_archive_and_local_content_are_verified(self):
        def encoded(value): return json.dumps(value).encode()
        content = {'manifest.json': encoded(dict(version='two-city-distance-v2')),
                   'status.json': encoded(dict(state='server-complete')),
                   'audit.json': encoded(dict(state='passed', unique_results=66, trajectory_records=231726)),
                   'registry.json': encoded({str(i): {} for i in range(66)})}
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
        content['delivery-files.json'] = encoded(dict(state='sealed', files=hashes,
                                                       manifest_sha256=hashes['manifest.json']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'test.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                for name, data in content.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            receipt = extract_verified(archive, root / 'out')
            self.assertEqual(receipt['verified_files'], 5)
            (root / 'out' / 'status.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                verify(root / 'out')

    def test_archive_symlinks_are_rejected_before_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'unsafe.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                member = tarfile.TarInfo('link')
                member.type = tarfile.SYMTYPE
                member.linkname = '/etc/passwd'
                tar.addfile(member)
            with self.assertRaisesRegex(RuntimeError, 'regular output files'):
                extract_verified(archive, root / 'out')
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
