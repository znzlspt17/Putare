"""상주 실행은 문제별 결과를 분리하고 프로세스를 재사용해야 한다."""
import unittest
import io
import subprocess
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from moe_prune.resident import ResidentExperiment, ResidentSession


class ResidentProtocolTests(unittest.TestCase):
    def test_two_requests_share_process_and_close_before_return(self):
        process = Mock()
        process.poll.return_value = 0
        process.stdout = io.StringIO('{"status":"ok"}\n{"status":"ok"}\n')
        process.wait.return_value = 0
        with patch('moe_prune.resident.spawn_native', return_value=process), \
             patch('moe_prune.resident.MemoryGuard'), \
             patch('moe_prune.resident.execution_guard'), \
             patch('moe_prune.resident.ProcessGpuMemorySampler') as sampler, \
             patch('moe_prune.resident.psutil.Process'):
            sampler.return_value.snapshot.return_value = {}
            session = ResidentSession('engine.exe', {}, {'model': 'model', 'n_gpu_layers': 0})
            session.request('one.json', 'one.jsonl')
            session.request('two.json', 'two.jsonl')
            session.close()
        self.assertEqual(process.stdin.write.call_count, 2)
        process.wait.assert_called()

    def test_invalid_protocol_response_stops_request(self):
        process = Mock()
        process.poll.return_value = 0
        process.stdout = io.StringIO('bad-response\n')
        with patch('moe_prune.resident.spawn_native', return_value=process), \
             patch('moe_prune.resident.MemoryGuard'), \
             patch('moe_prune.resident.execution_guard'), \
             patch('moe_prune.resident.ProcessGpuMemorySampler'):
            session = ResidentSession('engine.exe', {}, {'model': 'model', 'n_gpu_layers': 0})
            with self.assertRaisesRegex(RuntimeError, '응답'):
                session.request('one.json', 'one.jsonl')
            session.close()
        process.wait.assert_called()

    def test_monitor_failure_blocks_next_request(self):
        process = Mock()
        process.poll.return_value = 0
        process.stdout = io.StringIO('')
        with patch('moe_prune.resident.spawn_native', return_value=process), \
             patch('moe_prune.resident.MemoryGuard'), \
             patch('moe_prune.resident.execution_guard'), \
             patch('moe_prune.resident.ProcessGpuMemorySampler'):
            session = ResidentSession('engine.exe', {}, {'model': 'model', 'n_gpu_layers': 0})
            session.failure = MemoryError('guard exhausted')
            with self.assertRaisesRegex(MemoryError, 'guard exhausted'):
                session.request('one.json', 'one.jsonl')
            session.close()
        process.stdin.write.assert_not_called()

    def test_unconfirmed_exit_poisoned_before_lock_release(self):
        process = Mock()
        process.poll.return_value = None
        process.stdout = io.StringIO('')
        process.wait.side_effect = subprocess.TimeoutExpired('engine', 1)
        process.kill.side_effect = OSError('cannot kill')
        with patch('moe_prune.resident.spawn_native', return_value=process), \
             patch('moe_prune.resident.MemoryGuard'), \
             patch('moe_prune.resident.execution_guard') as lease, \
             patch('moe_prune.resident.ProcessGpuMemorySampler'), \
             patch('moe_prune.resident.psutil.Process'), \
             patch('moe_prune.resident.poison_execution') as poison:
            session = ResidentSession('engine.exe', {}, {'model': 'model', 'n_gpu_layers': 0})
            with self.assertRaisesRegex(RuntimeError, '종료'):
                session.close()
            poison.assert_called()
            lease.return_value.__exit__.assert_called()

    def test_experiment_keeps_one_session_and_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'model.gguf'; model.write_bytes(b'model')
            engine = root / 'engine.exe'; engine.write_bytes(b'engine')
            request = {'model': str(model), 'n_gpu_layers': 0, 'offload_kqv': False}
            def respond(req, out):
                Path(out).write_text(json.dumps({'type': 'result', 'text': 'ok'}) + '\n')
                return {'memory_last': {}}
            with patch('moe_prune.resident.ResidentSession') as session:
                session.return_value.request.side_effect = respond
                session.return_value.failure = None
                with ResidentExperiment(root / 'run', {'engine': str(engine)}) as exp:
                    self.assertEqual(exp.run('one', request)['text'], 'ok')
                    self.assertEqual(exp.run('two', request)['text'], 'ok')
                    self.assertEqual(session.call_count, 1)
                    model.write_bytes(b'changed model')
                    with self.assertRaisesRegex(ValueError, '변경'):
                        exp.run('three', request)
                session.return_value.close.assert_called_once()

    def test_oom_reaps_session_before_next_placement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'model.gguf'; model.write_bytes(b'model')
            engine = root / 'engine.exe'; engine.write_bytes(b'engine')
            request = {'model': str(model), 'n_gpu_layers': 24, 'offload_kqv': True}
            def respond(req, out):
                Path(out).write_text(json.dumps({'type': 'result', 'text': 'ok'}) + '\n')
                return {'memory_last': {}}
            with patch('moe_prune.resident.ResidentSession') as session:
                first, second = Mock(), Mock()
                first.request.side_effect = MemoryError('out of memory')
                first.failure = second.failure = None
                second.request.side_effect = respond
                def spawn(*args, **kwargs):
                    if session.call_count == 2:
                        first.close.assert_called_once()
                        return second
                    return first
                session.side_effect = spawn
                with ResidentExperiment(root / 'run', {'engine': str(engine)}) as exp:
                    with self.assertRaises(MemoryError):
                        exp.run('one', request)
                    result = exp.run('two', {**request, 'n_gpu_layers': 20})
                    self.assertEqual(result['text'], 'ok')
                second.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
