import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from moe_prune.gguf import GGUF, sha256
from test_gguf_integration import fixture

NOTEBOOK = Path(__file__).resolve().parents[1] / 'ipynb/03_expert_ordering.ipynb'


class OrderedNotebookTests(unittest.TestCase):
    def cells(self):
        self.assertTrue(NOTEBOOK.is_file(), '정렬 모델 생성용 03번 노트북이 필요합니다')
        nb = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(nb)
        return [cell.source for cell in nb.cells if cell.cell_type == 'code']

    def test_default_and_explicit_paths(self):
        import tools.reorder_experts as reorder
        cells = self.cells()
        namespace = {}
        with patch.object(reorder, 'export', create=True, return_value={'output_path': 'output', 'output_bytes': 1, 'output_sha256': 'test', 'verification': {}, 'limitation': 'unverified'}) as export:
            for cell in cells:
                exec(compile(cell, str(NOTEBOOK), 'exec'), namespace)
            export.assert_not_called()
            namespace.update(RUN_REORDER=True, ORIGINAL_MODEL=Path('source'), ORDERED_MODEL=Path('output'), ORDERED_MANIFEST=Path('manifest'))
            for cell in cells[1:]:
                exec(compile(cell, str(NOTEBOOK), 'exec'), namespace)
            export.assert_called_once_with(source=Path('source'), output=Path('output'), manifest=Path('manifest'))

    def test_export_preserves_and_verifies_all_experts(self):
        import tools.reorder_experts as reorder
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, manifest = root/'source.gguf', root/'out.gguf', root/'run/manifest.json'
            fixture(source, experts=128, top_k=8)
            before = source.read_bytes()
            counts = [[128-i for i in range(128)] for _ in range(30)]
            perms = {i: list(reversed(range(128))) for i in range(30)}
            original_architecture, original_rewrite = GGUF.architecture, GGUF.rewrite
            # 2026-11-09: 1층 합성 GGUF의 실제 바이트 재정렬을 수행하면서 진입 시 30층 계약만 대체한다.
            def rewrite_fixture(model, destination, *, permutation):
                with patch.object(GGUF, 'architecture', original_architecture):
                    return original_rewrite(model, destination, permutation={0: permutation[0]})
            with patch.object(reorder, 'profile', return_value=(counts, [10]*30, perms, {})), patch.object(reorder, 'SOURCE_HASH', sha256(source)), patch.object(GGUF, 'architecture', return_value=(list(range(30)), 128, 8)), patch.object(GGUF, 'rewrite', rewrite_fixture):
                partial = output.with_name(output.name + '.partial')
                partial.write_bytes(b'preserve')
                with self.assertRaises(FileExistsError):
                    reorder.export(source=source, output=output, manifest=manifest)
                self.assertEqual(partial.read_bytes(), b'preserve')
                self.assertFalse(output.exists())
                self.assertFalse(manifest.exists())
                partial.unlink()
                result = reorder.export(source=source, output=output, manifest=manifest)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(GGUF(output).architecture(), ([0], 128, 8))
            self.assertEqual(result['layers'][0]['new_to_old'], list(reversed(range(128))))
            self.assertTrue(result['verification']['all_tensor_bytes_verified'])
            self.assertEqual(json.loads(manifest.read_text(encoding='utf-8')), result)
            with self.assertRaises(FileExistsError):
                reorder.export(source=source, output=output, manifest=manifest)
            self.assertEqual(source.read_bytes(), before)

    def test_cli_passes_paths_to_export(self):
        import tools.reorder_experts as reorder
        with patch('sys.argv', ['reorder_experts.py', '--source', 'src', '--output', 'dst', '--manifest', 'map']), patch.object(reorder, 'export') as export:
            reorder.main()
            export.assert_called_once_with(source=Path('src'), output=Path('dst'), manifest=Path('map'))


if __name__ == '__main__':
    unittest.main()
