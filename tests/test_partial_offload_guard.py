"""부분 offload의 preflight 예산을 실제 GGUF tensor 정보로 확인한다."""
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_prune.memory_safety import GIB, MemoryGuard


def _string(value):
    data = value.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def _fixture(path):
    metadata = [('general.architecture', 8, _string('gemma4')),
                ('gemma4.block_count', 4, struct.pack('<I', 4)),
                ('gemma4.expert_count', 4, struct.pack('<I', 4)),
                ('gemma4.expert_used_count', 4, struct.pack('<I', 2))]
    tensors = [('token_embd.weight', 512), ('rope_freqs.weight', 64), ('output_norm.weight', 64)]
    for index, size in enumerate((1024, 2048, 4096, 8192)):
        tensors.extend([(f'blk.{index}.ffn_gate_inp.weight', 64), (f'blk.{index}.other.weight', size)])
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for name, kind, data in metadata:
        header += _string(name) + struct.pack('<I', kind) + data
    offset = 0
    for name, size in tensors:
        offset += -offset % 32
        header += _string(name) + struct.pack('<I', 1) + struct.pack('<Q', size // 4) + struct.pack('<IQ', 0, offset)
        offset += size
    with path.open('wb') as stream:
        stream.write(header)
        stream.write(b'\0' * (-stream.tell() % 32))
        for _, size in tensors:
            stream.write(b'\0' * (-stream.tell() % 32))
            stream.write(b'\0' * size)


class PartialOffloadGuardTests(unittest.TestCase):
    def test_partial_offload_accepts_capacity_for_selected_layers_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.gguf'
            _fixture(model)
            # 4 repeating + output: n_gpu_layers=2 selects blk.3 plus output.
            selected_payload = 8192 + 64
            non_block_payload = 512 + 64 + 64
            free = 2 * GIB + selected_payload + non_block_payload + 1024
            self.assertLess(free, 2 * GIB + model.stat().st_size)
            with patch('moe_prune.memory_safety._ram_sample', return_value=(16 * GIB, 8 * GIB, 'fixture')), \
                 patch('moe_prune.memory_safety.AdapterMemory') as adapter:
                adapter.return_value.sample.return_value = (4 * GIB, 4 * GIB - free)
                guard = MemoryGuard({}, {'model': str(model), 'n_gpu_layers': 2})
                try:
                    observed = guard.preflight()
                finally:
                    guard.close()
            self.assertGreaterEqual(observed['vram_estimated_model_bytes'], selected_payload + non_block_payload)
            self.assertLess(observed['vram_estimated_model_bytes'], model.stat().st_size)
            self.assertNotIn('vram_preflight_required_bytes', observed)

    def test_partial_offload_rejects_invalid_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'invalid.gguf'
            model.write_bytes(b'not GGUF')
            with patch('moe_prune.memory_safety._ram_sample', return_value=(16 * GIB, 8 * GIB, 'fixture')), \
                 patch('moe_prune.memory_safety.AdapterMemory') as adapter:
                adapter.return_value.sample.return_value = (8 * GIB, 0)
                guard = MemoryGuard({}, {'model': str(model), 'n_gpu_layers': 2})
                try:
                    with self.assertRaises((ValueError, EOFError)):
                        guard.preflight()
                finally:
                    guard.close()

    def test_partial_offload_warns_if_estimated_weights_exceed_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.gguf'
            _fixture(model)
            # token embedding도 output으로 GPU에 놓일 수 있으므로 전역 tensor를 포함한다.
            free = 128
            with patch('moe_prune.memory_safety._ram_sample', return_value=(16 * GIB, 8 * GIB, 'fixture')), \
                 patch('moe_prune.memory_safety.AdapterMemory') as adapter:
                adapter.return_value.sample.return_value = (4 * GIB, 4 * GIB - free)
                guard = MemoryGuard({}, {'model': str(model), 'n_gpu_layers': 2})
                try:
                    observed=guard.preflight()
                    self.assertTrue(any(x['kind']=='predicted' and x['resource']=='vram' for x in observed['warning_events']))
                finally:
                    guard.close()


if __name__ == '__main__':
    unittest.main()
