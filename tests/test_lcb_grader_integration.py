"""실제 WSL 공식 채점기를 격리 경로로만 실행하는 선택 통합 검사."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from moe_prune.benchmark_data import WSL_DISTRO, WSL_SOURCE, WSL_VENV, windows_to_wsl_path, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('RUN_WSL_LCB_TESTS') == '1', 'WSL 공식 채점기는 명시 실행')
class OfficialGraderTests(unittest.TestCase):
    def test_correct_wrong_and_timeout(self):
        tests = json.dumps([{'input': '3\n', 'output': '6\n', 'testtype': 'stdin'}])
        problem = {'public_test_cases': tests, 'private_test_cases': tests, 'metadata': '{}'}
        cases = [('print(int(input()) * 2)', True), ('print(9)', False), ('while True: pass', False)]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for i, (code, expected) in enumerate(cases):
                input_file, output_file = folder / f'in{i}.json', folder / f'out{i}.json'
                input_file.write_text(json.dumps({'id': str(i), 'problem': problem,
                                                  'text': f'```python\n{code}\n```'}), encoding='utf-8')
                command = ['wsl.exe', '--distribution', WSL_DISTRO, '--exec', 'bash',
                           windows_to_wsl_path(ROOT / 'tools/lcb_sandbox.sh'), WSL_SOURCE, WSL_VENV,
                           windows_to_wsl_path(input_file), windows_to_wsl_path(output_file)]
                subprocess.run(command, check=True)
                result = json.loads(output_file.read_text(encoding='utf-8'))
                self.assertIn('passed', result, result)
                self.assertEqual(result['passed'], expected)
                self.assertEqual(result['pass_at_1'], float(expected))
                if i == 2:
                    self.assertEqual(result['details']['error_code'], -3)

    def test_real_v6_problem_with_intentionally_wrong_code(self):
        snapshot = ROOT / 'runs/datasets/aime2026-lcbv6-new-001'
        if not (snapshot / 'manifest.json').is_file():
            self.skipTest('실제 공개 snapshot을 아직 준비하지 않았습니다')
        records, _ = load_snapshot(snapshot)
        record = next(row for row in records if row['domain'] == 'code')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            input_file, output_file = folder / 'input.json', folder / 'output.json'
            input_file.write_text(json.dumps({'id': record['id'], 'problem': record['lcb_problem'],
                                              'text': '```python\nraise RuntimeError()\n```'}), encoding='utf-8')
            subprocess.run(['wsl.exe', '--distribution', WSL_DISTRO, '--exec', 'bash',
                            windows_to_wsl_path(ROOT / 'tools/lcb_sandbox.sh'), WSL_SOURCE, WSL_VENV,
                            windows_to_wsl_path(input_file), windows_to_wsl_path(output_file)], check=True)
            result = json.loads(output_file.read_text(encoding='utf-8'))
            self.assertIn('passed', result, result)
            self.assertFalse(result['passed'])
            self.assertGreater(result['test_count'], 0)


if __name__ == '__main__':
    unittest.main()
