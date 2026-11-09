import ast
from pathlib import Path
import unittest

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'ipynb'


class ColabNotebookTests(unittest.TestCase):
    def sources(self, pattern):
        path, = NOTEBOOKS.glob(pattern)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        return [cell.source for cell in notebook.cells if cell.cell_type == 'code']

    def test_execution_flags_default_to_false(self):
        for path in sorted(NOTEBOOKS.glob('*.ipynb')):
            with self.subTest(notebook=path.name):
                cells = self.sources(path.name)
                flags = []
                for cell in cells:
                    tree = ast.parse(cell)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                            name = node.targets[0].id
                            if name.startswith(('RUN_', 'PREPARE_', 'BUILD_', 'DOWNLOAD_', 'VERIFY_')) or name in ('INSTALL_DEPENDENCIES', 'WRITE_MODELS'):
                                if isinstance(node.value, ast.Constant) and type(node.value.value) is bool:
                                    flags.append((name, node.value.value))
                self.assertTrue(flags)
                self.assertTrue(all(value is False for _, value in flags), flags)

    def test_cuda_paths_and_grading_entrypoints(self):
        one = '\n'.join(self.sources('01_*.ipynb'))
        two = '\n'.join(self.sources('02_*.ipynb'))
        four = '\n'.join(self.sources('04_*.ipynb'))
        five = '\n'.join(self.sources('05_*.ipynb'))
        for code in (one, two, four, five):
            self.assertNotIn('llama-expert-research.exe', code)
            self.assertNotIn('gfx1201', code)
            self.assertNotIn('WSL_', code)
            self.assertNotIn('grade_wsl', code)
        self.assertIn("'ram_usage_warn_fraction': .80", one)
        self.assertIn("'vram_usage_warn_fraction': .97", one)
        self.assertIn('prepare_grader(ROOT)', two)
        self.assertIn('grade_snapshot(', two)
        self.assertIn('work.grade(GENERATION_DIR)', four)
        self.assertIn('manual80.generate(', five)
        self.assertIn('manual80.grade(', five)

    def test_dependencies_are_colab_cuda_compatible(self):
        import tomllib
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        dependencies = config['project']['dependencies']
        self.assertTrue(any(value.startswith('nvidia-ml-py') for value in dependencies))
        self.assertFalse(any('rocm' in value.lower() or value.startswith('torch') for value in dependencies))
        self.assertNotIn('sources', config.get('tool', {}).get('uv', {}))


if __name__ == '__main__':
    unittest.main()
