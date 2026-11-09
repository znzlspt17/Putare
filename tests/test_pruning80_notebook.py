import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from moe_prune.gguf import GGUF, sha256
from test_gguf_integration import fixture


NOTEBOOK = Path(__file__).resolve().parents[1] / 'ipynb/05_expert_pruning_80.ipynb'


class Pruning80NotebookTests(unittest.TestCase):
    def cells(self):
        self.assertTrue(NOTEBOOK.is_file(), '104개에서 80개 모델을 생성하는 05번 노트북이 필요합니다')
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        return [cell.source for cell in notebook.cells if cell.cell_type == 'code']

    def test_run_all_defaults_do_not_export(self):
        import tools.prune_e80 as work
        with patch.object(work, '_export_e80', side_effect=AssertionError('unexpected export')) as export:
            namespace = {}
            for cell in self.cells():
                exec(compile(cell, str(NOTEBOOK), 'exec'), namespace)
            export.assert_not_called()
            self.assertEqual(namespace['SOURCE'], work.SOURCE)
            self.assertEqual(namespace['OUTPUT'], work.OUTPUT)
            self.assertEqual(namespace['RUN'], work.RUN)
            self.assertEqual(namespace['SOURCE_MANIFEST'], work.PARENT)
            self.assertEqual(namespace['REORDER_MANIFEST'], work.REORDER)
            self.assertEqual(namespace['EXPECTED_SHA'], work.SOURCE_SHA)
            self.assertEqual(namespace['EXPECTED_BYTES'], work.SOURCE_BYTES)
            self.assertEqual(namespace['EXPECTED_LAYERS'], list(range(30)))

    def test_export_preserves_first_80_experts(self):
        cells = self.cells()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, run = root / 'source.gguf', root / 'output.gguf', root / 'run'
            fixture(source, experts=104, top_k=8)
            reorder = root / 'reorder.json'
            reorder.write_text(json.dumps({'output_sha256': 'parent', 'layers': [{'layer': 0, 'new_to_old': list(range(128))}]}), encoding='utf-8')
            parent = root / 'parent.json'
            parent.write_text(json.dumps({'source_sha256': 'parent', 'mapping_manifest_sha256': sha256(reorder), 'output_sha256': sha256(source), 'expert_count': 104, 'top_k': 8, 'file_validation_status': 'passed', 'layers': [{'layer': 0, 'pruned_to_ordered': list(range(104)), 'pruned_to_original': list(range(104))}]}), encoding='utf-8')
            namespace = {}
            exec(compile(cells[0], str(NOTEBOOK), 'exec'), namespace)
            namespace.update(SOURCE=source, OUTPUT=output, RUN=run, SOURCE_MANIFEST=parent,
                             REORDER_MANIFEST=reorder, EXPECTED_SHA=sha256(source),
                             EXPECTED_BYTES=source.stat().st_size, EXPECTED_LAYERS=[0], RUN_EXPORT=True)
            for cell in cells[1:]:
                exec(compile(cell, str(NOTEBOOK), 'exec'), namespace)
            self.assertEqual(GGUF(output).architecture(), ([0], 80, 8))
            result = json.loads((run / 'export-manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(result['layers'][0]['e80_to_source104'], list(range(80)))
            self.assertTrue(result['verification']['all_tensor_bytes_verified'])
            self.assertEqual(result['inference_status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
