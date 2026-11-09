"""비격리 채점 worker의 출력 경계와 최소 환경을 검증한다."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_prune.grading import read_result, worker_environment


class WorkerContractTests(unittest.TestCase):
    def test_worker_environment_does_not_inherit_unlisted_values(self):
        with patch.dict(os.environ, {'HF_TOKEN': 'fake-test-value'}):
            environment = worker_environment()
        self.assertNotIn('HF_TOKEN', environment)
        self.assertEqual(environment['HOME'], '/tmp')

    def test_result_symlink_and_oversize_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_bytes(b'12345')
            with self.assertRaises(ValueError):
                read_result(path, maximum_bytes=4)
            self.assertEqual(read_result(path, maximum_bytes=5), '12345')
            target = Path(directory) / 'elsewhere'
            target.write_text('{}', encoding='utf-8')
            path.unlink()
            try:
                path.symlink_to(target)
            except OSError:
                self.skipTest('symlink 권한 없음')
            with self.assertRaises((OSError, ValueError)):
                read_result(path)
