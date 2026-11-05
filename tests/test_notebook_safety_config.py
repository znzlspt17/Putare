"""노트북은 시간 제한 대신 시스템 메모리 여유를 기본 보호 기준으로 사용한다."""
import ast
import json
from pathlib import Path
import unittest


class NotebookSafetyConfigTests(unittest.TestCase):
    def test_notebook_and_generator_use_memory_headroom_defaults(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'ipynb/01_Gemma4_Expert_Pruning_연구.ipynb').read_text(encoding='utf-8'))
        cell = next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and 'INSTALL_DEPENDENCIES = False' in ''.join(c['source']))
        config = next(n.value for n in ast.parse(cell).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets))
        values = {k.value: v for k, v in zip(config.keys, config.values)}
        expected = {'ram_budget_bytes': None, 'min_available_ram_bytes': 4 * 1024**3,
                    'min_available_commit_bytes': 4 * 1024**3, 'min_free_vram_bytes': 1024**3,
                    'vram_workspace_bytes': 1024**3}
        for key, value in expected.items():
            self.assertIn(key, values)
            self.assertEqual(eval(compile(ast.Expression(values[key]), '<config>', 'eval'), {'__builtins__': {}}), value)
        generator = (root / 'tools/build_research_notebook.py').read_text(encoding='utf-8')
        self.assertIn(cell, generator)


if __name__ == '__main__':
    unittest.main()
