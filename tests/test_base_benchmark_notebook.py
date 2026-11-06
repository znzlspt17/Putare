"""02번 노트북의 안전 기본값과 실제 셀 실행 계약."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class BaseBenchmarkNotebookTests(unittest.TestCase):
    def test_default_run_all_has_no_network_subprocess_or_execution(self):
        notebook = nbformat.read(ROOT / 'ipynb/02_base_model_benchmark.ipynb', as_version=4)
        nbformat.validate(notebook)
        namespace = {'__name__': '__notebook_default_test__'}
        with patch('subprocess.run', side_effect=AssertionError('unexpected subprocess')), \
             patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network')), \
             patch('urllib.request.build_opener', side_effect=AssertionError('unexpected network')), \
             patch('moe_prune.base_benchmark.BenchmarkRun.run', side_effect=AssertionError('unexpected inference')), \
             patch('moe_prune.benchmark_data.grade_snapshot', side_effect=AssertionError('unexpected grading')), \
             contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    exec(compile(cell.source, f'cell-{index}', 'exec'), namespace)
        for key in ('PREPARE_WSL', 'PREPARE_ENGINE', 'BUILD_ENGINE', 'DOWNLOAD_DATA', 'VERIFY_DATA_IDS',
                    'RUN_SMOKE', 'RUN_BENCHMARK', 'RUN_GRADING'):
            self.assertFalse(namespace[key], key)
        self.assertEqual(namespace['CONFIG']['context_size'], 65535)
        self.assertEqual(namespace['CONFIG']['max_tokens'], 32768)


if __name__ == '__main__':
    unittest.main()
