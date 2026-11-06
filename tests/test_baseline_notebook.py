"""Pruning 없이 원본 모델의 평가·속도 측정 경로를 검증한다."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import subprocess

import nbformat

from moe_prune.gguf import sha256
from moe_prune.research import baseline_environment, canonical_hash, write_json, write_jsonl
from test_gguf_integration import fixture
from test_notebook_flow import SyntheticNative


class BaselineNotebookTests(unittest.TestCase):
    def test_baseline_runs_without_pruning_or_writes(self):
        repository = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(repository / 'ipynb/01_Gemma4_Expert_Pruning_연구.ipynb', as_version=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, engine, data = root / 'model.gguf', root / 'engine.exe', root / 'data'
            fixture(model, experts=128, top_k=8)
            engine.write_bytes(b'synthetic')
            groups = {split: [
                {'id': f'{split}:{domain}', 'domain': domain, 'dataset': domain,
                 'prompt': f'{split} {domain} question', 'reference': '#### 42' if domain == 'math' else 'return 42'}
                for domain in ('math', 'code', 'general')]
                for split in ('calibration', 'selection', 'evaluation')}
            for split, rows in groups.items():
                write_jsonl(data / f'{split}.jsonl', rows)
            write_json(data / 'manifest.json', {
                'file_sha256': {key: sha256(data / f'{key}.jsonl') for key in groups},
                'full_record_hashes': {key: canonical_hash(value) for key, value in groups.items()}})
            namespace = {'__name__': '__baseline_synthetic_test__'}
            captured = io.StringIO()
            with patch('moe_prune.research.source_revision', return_value={'git_commit': 'synthetic', 'git_dirty': False}), \
                 patch('moe_prune.research.baseline_environment', return_value={
                     'gpu': [{'name': 'synthetic GPU', 'driver_version': 'synthetic'}], 'ram_bytes': 123}, create=True), \
                 patch('moe_prune.research.subprocess.Popen', side_effect=SyntheticNative), \
                 patch('moe_prune.memory_safety._available_memory', return_value=(64 * 1024**3, 64 * 1024**3)), \
                 patch('moe_prune.memory_safety.AdapterMemory') as adapter, \
                 patch('psutil.Process'), contextlib.redirect_stdout(captured):
                adapter.return_value.sample.return_value = (16 * 1024**3, 0)
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type != 'code':
                        continue
                    exec(compile(cell.source, f'cell-{index}', 'exec'), namespace)
                    if 'INSTALL_DEPENDENCIES = False' in cell.source:
                        self.assertFalse(namespace['RUN_PRUNING_ANALYSIS'])
                        namespace.update(ROOT=root, MODEL=model, ENGINE=engine, DATA=data,
                                         RUN_MODEL=True, WRITE_MODELS=False, RUN_EVALUATION=True, RUN_BENCHMARK=True,
                                         RUN_PRUNING_ANALYSIS=False)
                        namespace['CONFIG']['engine'] = str(engine)
            run = namespace['exp'].root
            self.assertEqual(namespace['exports'], {})
            self.assertEqual(set(namespace['evaluations']), {'original'})
            self.assertEqual(set(namespace['speed_results']), {'original'})
            self.assertEqual(json.loads((run / 'paired-quality.json').read_text(encoding='utf-8')), {})
            self.assertIn('비교 대상 없음', captured.getvalue())
            manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
            frozen = manifest['stages']['evaluation_frozen']
            self.assertEqual(frozen['masks'], {})
            self.assertEqual(frozen['mask_sha256'], canonical_hash({}))
            self.assertEqual(frozen['records_sha256'], canonical_hash(groups['evaluation']))
            for name in ('evaluation-original.json', 'evaluation-original-analysis.json',
                         'sanity-original.json', 'tradeoff-points.json', 'conclusion-review.json'):
                self.assertTrue((run / name).is_file(), name)
            self.assertEqual(namespace['conclusion']['observed_file_sizes']['original'], model.stat().st_size)
            self.assertIn('evaluation-original-analysis.json', namespace['conclusion']['evidence'])
            summary = json.loads((run / 'baseline-summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['environment']['gpu'][0]['name'], 'synthetic GPU')
            self.assertEqual(summary['model']['sha256'], sha256(model))
            self.assertEqual(summary['dataset']['manifest_sha256'], sha256(data / 'manifest.json'))
            self.assertEqual(summary['sampling']['temperature'], 0)
            self.assertEqual(summary['sampling']['evaluation_max_tokens'], 8192)
            self.assertEqual(summary['quality']['summary']['math']['numeric_math_accuracy'], 1)
            self.assertIsNone(summary['quality']['summary']['code']['code_execution_pass_rate'])
            self.assertIn('speed-original-benchmark.json', summary['evidence'])
            self.assertEqual(summary['speed'], namespace['speed_results']['original'])
            self.assertFalse((run / 'models').exists())
            self.assertFalse((run / 'candidate-masks.json').exists())
            jobs = [p.name for p in (run / 'jobs').iterdir()]
            self.assertFalse(any(name.startswith(('profile-', 'instrumentation-', 'selection-loss-', 'sensitivity-')) for name in jobs))

    def test_write_models_does_not_override_disabled_pruning(self):
        repository = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(repository / 'ipynb/01_Gemma4_Expert_Pruning_연구.ipynb', as_version=4)
        source = next(c.source for c in notebook.cells if c.cell_type == 'code' and "models/noop.gguf" in c.source)
        with tempfile.TemporaryDirectory() as directory:
            exp, original = Mock(), Mock()
            namespace = {'ROOT': Path(directory), 'RUN_ID': 'baseline', 'RUN_MODEL': True,
                         'RUN_PRUNING_ANALYSIS': False, 'WRITE_MODELS': True, 'gate_records': [{}],
                         'exp': exp, 'original': original}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, 'noop-cell', 'exec'), namespace)
            exp.require.assert_not_called()
            original.rewrite.assert_not_called()

    def test_original_speed_does_not_require_evaluation_or_exports(self):
        repository = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(repository / 'ipynb/01_Gemma4_Expert_Pruning_연구.ipynb', as_version=4)
        source = next(c.source for c in notebook.cells if c.cell_type == 'code' and 'speed_results = {}' in c.source)
        benchmark = Mock(return_value={'speed': 42})
        namespace = {'RUN_MODEL': True, 'RUN_BENCHMARK': True, 'gate_records': [{'id': 'x'}],
                     'exports': {}, 'MODEL': Path('model.gguf'), 'exp': Mock(), 'benchmark': benchmark}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, 'speed-cell', 'exec'), namespace)
        self.assertEqual(namespace['speed_results'], {'original': {'speed': 42}})
        benchmark.assert_called_once()

    def test_gpu_metadata_probe_timeout_is_unknown(self):
        with patch('moe_prune.research.platform.system', return_value='Windows'), \
             patch('moe_prune.research.subprocess.run', side_effect=subprocess.TimeoutExpired('probe', 10)):
            observed = baseline_environment()
            self.assertIsNone(observed['gpu'])
            self.assertIn('physical_cores', observed['cpu'])
            self.assertIn('logical_cores', observed['cpu'])

    def test_gpu_metadata_probe_decodes_single_or_multiple_adapters(self):
        for rows in ({'Name': 'GPU A', 'DriverVersion': '1.2'},
                     [{'Name': 'GPU A', 'DriverVersion': '1.2'}, {'Name': 'GPU B', 'DriverVersion': None}]):
            with self.subTest(rows=rows), patch('moe_prune.research.platform.system', return_value='Windows'), \
                 patch('moe_prune.research.subprocess.run', return_value=Mock(stdout=json.dumps(rows))):
                expected = rows if isinstance(rows, list) else [rows]
                self.assertEqual(baseline_environment()['gpu'],
                                 [{'name': r['Name'], 'driver_version': r['DriverVersion']} for r in expected])


if __name__ == '__main__':
    unittest.main()
