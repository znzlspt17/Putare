"""노트북의 Colab 시스템 메모리 경고 기준을 확인한다."""
import ast
import json
from pathlib import Path
import unittest


class NotebookSafetyConfigTests(unittest.TestCase):
    def test_notebook_uses_memory_headroom_defaults(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'ipynb/01_base_model_분석.ipynb').read_text(encoding='utf-8'))
        cell = next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and 'INSTALL_DEPENDENCIES = False' in ''.join(c['source']))
        config = next(n.value for n in ast.parse(cell).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets))
        values = {k.value: v for k, v in zip(config.keys, config.values)}
        expected = {'ram_usage_warn_fraction': .80, 'vram_usage_warn_fraction': .97,
                    'n_gpu_layers': 31}
        for key, value in expected.items():
            self.assertIn(key, values)
            self.assertEqual(eval(compile(ast.Expression(values[key]), '<config>', 'eval'), {'__builtins__': {}}), value)


if __name__ == '__main__':
    unittest.main()
