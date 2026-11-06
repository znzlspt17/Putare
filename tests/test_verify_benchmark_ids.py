"""큰 JSONL의 ID 스트리밍과 chunk 경계 처리."""
import hashlib
import io
import unittest

from tools.verify_benchmark_ids import scan_stream


class VerifyIdsTests(unittest.TestCase):
    def test_stream_crosses_chunk_boundaries_without_duplicate_matches(self):
        raw = (b'{"value":"one", "question_id":"abc-1"}\n'
               b'{"value":"a fake \\\"question_id\\\":\\\"x\\\"", "question_id":"z-2"}\n')
        ids, rows, digest = scan_stream(io.BytesIO(raw), chunk_size=7)
        self.assertEqual(ids, {'abc-1', 'z-2'})
        self.assertEqual(rows, 2)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_duplicate_or_missing_ids_fail_closed(self):
        for raw in (b'{"question_id":"a"}\n{"question_id":"a"}\n',
                    b'{"question_id":"a"}\n{"x":1}\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                scan_stream(io.BytesIO(raw), chunk_size=9)


if __name__ == '__main__':
    unittest.main()
