"""02번 평가의 외부 관측 계약을 합성 입력으로 검증한다."""
import unittest
import tempfile
import gzip
import json
from pathlib import Path
from unittest.mock import patch
from moe_prune.gguf import sha256

from moe_prune.base_benchmark import BenchmarkRun, aggregate_profiles, build_request, load_completed, run_with_offload, validate_routing, verify_runtime
from moe_prune.research import canonical_hash, write_json, write_jsonl


def event(layer, phase, index, expert=0):
    return {'type': 'routing', 'layer': layer, 'phase': phase, 'token_index': index,
            'expert_ids': [expert, (expert + 1) % 4], 'weights': [.75, .25],
            'weighted_output_norms': [1.25, .125]}


class BenchmarkTests(unittest.TestCase):
    def test_full_run_rejects_changed_grader_before_loading_model(self):
        records = [{'id': f'{domain}:{i}', 'domain': domain, 'dataset': domain, 'prompt': 'x'}
                   for domain, count in [('math', 30), ('code', 175)] for i in range(count)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('moe_prune.benchmark_data.load_snapshot', return_value=(records, {'grader_identity': {'tools/lcb_grade.py': 'old'}})), \
                 patch('moe_prune.benchmark_data.validate_id_proof', return_value={}), \
                 patch('moe_prune.benchmark_data.grader_identity', return_value={'tools/lcb_grade.py': 'new'}), \
                 patch('moe_prune.base_benchmark.GGUF', side_effect=AssertionError('model opened')):
                with self.assertRaisesRegex(ValueError, '채점'):
                    BenchmarkRun({}, root / 'model', root / 'engine', root / 'snapshot', root / 'run').run()

    def test_full_run_identity_includes_verified_v5_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'records.jsonl').write_text('records')
            (snapshot / 'manifest.json').write_text('{}')
            model, engine = root / 'model.gguf', root / 'engine.exe'
            model.write_bytes(b'model')
            engine.write_bytes(b'engine')
            records = [{'id': f'{domain}:{i}', 'domain': domain, 'dataset': domain,
                        'prompt': 'x', 'reference': '0'}
                       for domain, count in [('math', 30), ('code', 175)] for i in range(count)]
            class FakeModel:
                def architecture(self): return list(range(30)), 128, 8
            proof = {'status': 'verified', 'v5_count': 880, 'v6_new_count': 175}
            old_manifest={'grader_identity': {'tools/lcb_grade.py':'same', 'tools/lcb_sandbox.py':'old'},
                          'official_identity': {'source_digest':'trusted','python':'3.12.3'}}
            current_grader={'tools/lcb_grade.py':'same','moe_prune/grading.py':'new','execution_mode':'unisolated'}
            current_official={'source_digest':'trusted','python':'3.12.9'}
            with patch('moe_prune.benchmark_data.load_snapshot', return_value=(records, old_manifest)), \
                 patch('moe_prune.benchmark_data.validate_id_proof', return_value=proof), \
                 patch('moe_prune.benchmark_data.grader_identity', return_value=current_grader), \
                 patch('moe_prune.benchmark_data.official_identity', return_value=current_official), \
                 patch('moe_prune.base_benchmark.GGUF', return_value=FakeModel()), \
                 patch('moe_prune.base_benchmark.ResidentExperiment', side_effect=RuntimeError('stop after identity')):
                with self.assertRaisesRegex(RuntimeError, 'stop after identity'):
                    BenchmarkRun({}, model, engine, snapshot, root / 'run').run()
            identity = json.loads((root / 'run' / 'benchmark-identity.json').read_text(encoding='utf8'))
            self.assertEqual(identity['lcb_v5_id_proof'], canonical_hash(proof))
            self.assertEqual(identity['grader_identity'], current_grader)
            self.assertEqual(identity['official_identity'], current_official)

    def test_full_run_requires_v5_id_proof_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [{'id': f'{domain}:{i}', 'domain': domain, 'dataset': domain,
                        'prompt': 'x', 'reference': '0'}
                       for domain, count in [('math', 30), ('code', 175)] for i in range(count)]
            run_dir = root / 'run'
            with patch('moe_prune.benchmark_data.load_snapshot', return_value=(records, {})), \
                 patch('moe_prune.benchmark_data.validate_id_proof', side_effect=FileNotFoundError('prepare proof')) as proof, \
                 patch('moe_prune.base_benchmark.GGUF', side_effect=AssertionError('model loaded before proof')) as model:
                with self.assertRaisesRegex(FileNotFoundError, 'prepare proof'):
                    BenchmarkRun({}, root / 'model.gguf', root / 'engine.exe', root / 'snapshot', run_dir).run()
                proof.assert_called_once_with(root / 'snapshot')
                model.assert_not_called()
                self.assertFalse(run_dir.exists())

    def test_actual_cache_and_flash_must_be_observed_in_engine_log(self):
        request = {'context_size': 65535, 'offload_kqv': True}
        result = {'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0', 'actual_context_size': 65536,
                  'instrumentation': True, 'flash_attention_requested': True,
                  'offload_kqv_requested': True, 'seed': 42,
                  'sampling': {'strategy': 'sampled', 'temperature': 1.0, 'top_p': .95, 'top_k': 64}}
        log = ('llama_context: flash_attn            = enabled\n'
               'load_tensors: offloaded 16/31 layers to GPU\n'
               'load_tensors: ROCm0 model buffer size = 10.00 MiB\n'
               'llama_kv_cache: size = 100 MiB K (q8_0): 50 MiB, V (q8_0): 50 MiB\n')
        observed = verify_runtime(result, log, request)
        self.assertTrue(observed['flash_attention'])
        self.assertEqual(observed['offloaded_layers'], {'gpu': 16, 'total': 31})
        self.assertEqual(observed['model_buffers_mib'], [{'backend': 'ROCm0', 'mib': 10.0}])
        with self.assertRaises(ValueError):
            verify_runtime(result, log.replace('enabled', 'disabled'), request)
        with self.assertRaises(ValueError):
            verify_runtime(result, log.replace('V (q8_0)', 'V (f16)'), request)
        with self.assertRaises(ValueError):
            verify_runtime({**result, 'sampling': {**result['sampling'], 'top_k': 40}}, log, request)

    def test_two_problem_smoke_resumes_without_rerunning_completed_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            records = [{'id': f'{domain}:{i}', 'domain': domain, 'dataset': domain, 'prompt': 'x', 'reference': '0'}
                       for domain, count in [('math', 30), ('code', 175)] for i in range(count)]
            write_jsonl(snapshot / 'records.jsonl', records)
            write_json(snapshot / 'manifest.json', {'file_sha256': {'records': sha256(snapshot / 'records.jsonl')}})
            model, engine = root / 'model.gguf', root / 'engine.exe'
            model.write_bytes(b'model')
            engine.write_bytes(b'engine')
            calls = []
            class FakeModel:
                def architecture(self): return list(range(30)), 128, 8
            class FakeExperiment:
                def __init__(self, path, config): self.root = Path(path)
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def check(self): pass
                def run(self, name, request):
                    calls.append(request)
                    job = self.root / 'jobs' / name
                    job.mkdir(parents=True)
                    rows = [{'type': 'routing', 'layer': layer, 'phase': 'prefill', 'token_index': 0,
                             'expert_ids': list(range(8)), 'weights': [.125] * 8,
                             'weighted_output_norms': [1.] * 8} for layer in range(30)]
                    write_jsonl(job / 'events.jsonl', rows)
                    (job / 'stderr.log').write_text('llama_context: flash_attn = enabled\nllama_kv_cache: K (q8_0): 1 V (q8_0): 1\n')
                    return {'text': 'hello', 'token_ids': [42], 'stop_reason': 'length', 'prompt_tokens': 1,
                            'processed_decode_tokens': 0, 'routing_records': 30,
                            'instrumentation': True, 'flash_attention_requested': True,
                            'offload_kqv_requested': request['offload_kqv'],
                            'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0',
                            'actual_context_size': 65536,
                            'seed': 42, 'sampling': {'strategy': 'sampled', 'temperature': 1.0,
                                                     'top_p': .95, 'top_k': 64}}
            with patch('moe_prune.base_benchmark.GGUF', return_value=FakeModel()), \
                 patch('moe_prune.benchmark_data.load_snapshot', return_value=(records, {})), \
                 patch('moe_prune.benchmark_data.validate_id_proof', side_effect=AssertionError('smoke should skip proof')), \
                 patch('moe_prune.base_benchmark.ResidentExperiment', FakeExperiment):
                runner = BenchmarkRun({}, model, engine, snapshot, root / 'run')
                self.assertEqual(runner.run(smoke=True)['completed'], 2)
                self.assertEqual(len(calls), 2)
                self.assertEqual(runner.run(smoke=True)['completed'], 2)
                self.assertEqual(len(calls), 2)

    def test_resume_rejects_modified_routing_or_run_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with gzip.open(root / 'routing.gz', 'wt', encoding='utf8') as stream:
                stream.write('{"type":"routing"}\n')
            answer = {'id': 'q', 'identity_hash': 'config-a', 'routing_path': 'routing.gz',
                      'routing_sha256': sha256(root / 'routing.gz')}
            answer['answer_sha256'] = canonical_hash(answer)
            (root / 'answer.json').write_text(json.dumps(answer), encoding='utf8')
            self.assertEqual(load_completed(root / 'answer.json', 'q', 'config-a', root)['id'], 'q')
            with self.assertRaises(ValueError):
                load_completed(root / 'answer.json', 'q', 'config-b', root)
            (root / 'answer.json').write_text(json.dumps({**answer, 'text': 'changed'}), encoding='utf8')
            with self.assertRaises(ValueError):
                load_completed(root / 'answer.json', 'q', 'config-a', root)
            (root / 'answer.json').write_text(json.dumps(answer), encoding='utf8')
            (root / 'routing.gz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                load_completed(root / 'answer.json', 'q', 'config-a', root)

    def test_allocation_failure_does_not_change_placement(self):
        seen = []
        def invoke(placement):
            seen.append(placement)
            raise MemoryError('allocation')
        with self.assertRaises(MemoryError): run_with_offload(invoke)
        self.assertEqual(seen, [(31, True)])
        with self.assertRaisesRegex(RuntimeError, 'corrupt'):
            run_with_offload(lambda _: (_ for _ in ()).throw(RuntimeError('corrupt')))

    def test_request_always_profiles_without_exposing_answer(self):
        record = {'id': 'q', 'domain': 'math', 'prompt': 'question', 'reference': '123',
                  'system_prompt': 'Solve carefully'}
        request = build_request(record, 'original.gguf', (20, False), {})
        self.assertEqual(request['mode'], 'profile')
        self.assertTrue(request['instrumentation'])
        self.assertNotIn('reference', request)
        self.assertEqual(request['sampling'], {'temperature': 1.0, 'top_p': .95, 'top_k': 64})
        self.assertEqual((request['n_gpu_layers'], request['offload_kqv']), (20, False))
        self.assertEqual((request['cache_type_k'], request['cache_type_v']), ('q8_0', 'q8_0'))
        self.assertTrue(request['flash_attention'])
        self.assertTrue(request['all_token_logits'])
        self.assertEqual(request['system_prompt'], 'Solve carefully')

    def test_optional_fusion_and_routing_mode_are_explicit(self):
        record = {'id': 'q', 'prompt': 'question'}
        old = build_request(record, 'original.gguf', (31, True), {})
        self.assertNotIn('fusion_enabled', old)
        self.assertNotIn('instrumentation_level', old)
        request = build_request(record, 'original.gguf', (31, True),
                                {'fusion_enabled': True, 'instrumentation_level': 'routing',
                                 'context_size': 262144, 'max_tokens': 65536})
        self.assertIs(request['fusion_enabled'], True)
        self.assertEqual(request['instrumentation_level'], 'routing')
        self.assertEqual((request['context_size'], request['max_tokens']), (262144, 65536))

    def test_lightweight_routing_requires_all_positions_and_valid_normalized_gates(self):
        events = [{k: v for k, v in event(layer, 'prefill', token).items() if k != 'weighted_output_norms'}
                  for token in range(2) for layer in range(2)]
        events += [{k: v for k, v in event(layer, 'decode', 2).items() if k != 'weighted_output_norms'}
                   for layer in range(2)]
        result = {'prompt_tokens': 2, 'processed_decode_tokens': 1, 'instrumentation_level': 'routing'}
        self.assertEqual(validate_routing(iter(events), result, layers=2, experts=4, top_k=2), 6)
        for bad in (events[:-1], events + [events[0]], [{**events[0], 'weights': [.8, .8]}] + events[1:],
                    [{**events[0], 'expert_ids': [1, 1]}] + events[1:]):
            with self.subTest(bad=bad[:1]), self.assertRaises(ValueError):
                validate_routing(iter(bad), result, layers=2, experts=4, top_k=2)
        with self.assertRaises(ValueError):
            validate_routing(iter(events), {**result, 'instrumentation_level': 'full'}, layers=2, experts=4, top_k=2)

    def test_runtime_rejects_fusion_or_observer_policy_mismatch(self):
        request = {'context_size': 262144, 'offload_kqv': True, 'fusion_enabled': True,
                   'instrumentation_level': 'routing'}
        result = {'actual_context_size': 262144, 'instrumentation': True, 'flash_attention_requested': True,
                  'offload_kqv_requested': True, 'seed': 42, 'gpu_fusion_disabled': False,
                  'fusion_enabled': True, 'instrumentation_level': 'routing',
                  'sampling': {'strategy': 'sampled', 'temperature': 1.0, 'top_p': .95, 'top_k': 64}}
        log = 'llama_context: flash_attn = enabled\nllama_kv_cache: K (q8_0): 1 V (q8_0): 1\n'
        verify_runtime(result, log, request)
        for altered in ({'fusion_enabled': False}, {'gpu_fusion_disabled': True}, {'instrumentation_level': 'full'}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                verify_runtime({**result, **altered}, log, request)

    def test_validation_requires_each_layer_and_token_once(self):
        events = [event(layer, 'prefill', token) for token in range(2) for layer in range(2)]
        events += [event(layer, 'decode', 2) for layer in range(2)]
        result = {'prompt_tokens': 2, 'processed_decode_tokens': 1}
        self.assertEqual(validate_routing(iter(events), result, layers=2, experts=4, top_k=2), 6)
        with self.assertRaises(ValueError):
            validate_routing(iter(events[:-1]), result, layers=2, experts=4, top_k=2)
        with self.assertRaises(ValueError):
            validate_routing(iter(events + [events[0]]), result, layers=2, experts=4, top_k=2)
        with self.assertRaises(ValueError):
            validate_routing(iter(events), {**result, 'token_ids': [1, 2, 3], 'stop_reason': 'length'},
                             layers=2, experts=4, top_k=2)

    def test_per_problem_mean_does_not_weight_long_questions_more(self):
        samples = [('math', [event(0, 'prefill', i, 0) for i in range(3)]),
                   ('math', [event(0, 'prefill', 0, 2)])]
        profile = aggregate_profiles(samples, layers=1, experts=4)['math']['prefill']['0']
        self.assertEqual(profile['counts'][0], 3)
        self.assertEqual(profile['tokens'], 4)
        self.assertAlmostEqual(profile['token_rate'][0], .75)
        self.assertAlmostEqual(profile['problem_mean_rate'][0], .5)
