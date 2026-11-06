"""기준선 실행의 GPU 예산은 실제 adapter 사용량을 기준으로 한다."""
import unittest
from unittest.mock import patch

from moe_prune.memory_safety import GIB, MemoryGuard


class BenchmarkMemoryTests(unittest.TestCase):
    def test_cpu_kv_offload_is_accounted_in_host_memory(self):
        class FakeModel:
            metadata = {'general.architecture': 'gemma4', 'gemma4.block_count': 1,
                        'gemma4.attention.head_count_kv': [2],
                        'gemma4.attention.key_length': 64, 'gemma4.attention.value_length': 64,
                        'gemma4.attention.sliding_window_pattern': [False]}
        config = {'min_available_ram_bytes': 4 * GIB, 'min_available_commit_bytes': 4 * GIB,
                  'min_free_vram_bytes': GIB, 'vram_workspace_bytes': GIB}
        request = {'model': 'fake.gguf', 'n_gpu_layers': 0, 'context_size': 2048,
                   'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0', 'offload_kqv': False}
        with patch('moe_prune.memory_safety._available_memory', return_value=(8 * GIB, 8 * GIB)), \
             patch('moe_prune.gguf.GGUF', return_value=FakeModel()):
            observed = MemoryGuard(config, request).preflight()
            self.assertEqual(observed['ram_estimated_kv_bytes'], 2 * 2048 * 136)

    def test_preflight_accounts_for_quantized_attention_cache(self):
        class FakeModel:
            metadata = {'general.architecture': 'gemma4', 'gemma4.block_count': 2,
                        'gemma4.attention.head_count_kv': [2, 2],
                        'gemma4.attention.key_length': 64, 'gemma4.attention.value_length': 64,
                        'gemma4.attention.key_length_swa': 64, 'gemma4.attention.value_length_swa': 64,
                        'gemma4.attention.sliding_window': 128,
                        'gemma4.attention.sliding_window_pattern': [True, False],
                        'gemma4.attention.shared_kv_layers': 0}
        config = {'min_available_ram_bytes': 4 * GIB, 'min_available_commit_bytes': 4 * GIB,
                  'min_free_vram_bytes': GIB, 'vram_workspace_bytes': GIB}
        request = {'model': 'fake.gguf', 'n_gpu_layers': 3, 'context_size': 2048,
                   'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0', 'offload_kqv': True}
        with patch('moe_prune.memory_safety._available_memory', return_value=(32 * GIB, 32 * GIB)), \
             patch('moe_prune.memory_safety.AdapterMemory') as adapter, \
             patch('moe_prune.memory_safety._gpu_weight_estimate', return_value=100), \
             patch('moe_prune.gguf.GGUF', return_value=FakeModel()):
            adapter.return_value.sample.return_value = (16 * GIB, 0)
            observed = MemoryGuard(config, request).preflight()
            # pinned llama.cpp의 swa_full 기본값은 true이므로 두 계층 모두 전체 context를 예약한다.
            self.assertEqual(observed['vram_estimated_kv_bytes'], 2 * 2 * 2048 * 136)
            self.assertEqual(observed['vram_preflight_required_bytes'],
                             100 + observed['vram_estimated_kv_bytes'] + 2 * GIB)

    def test_total_usage_cannot_exceed_fraction_even_with_one_gib_free(self):
        config = {'min_available_ram_bytes': 4 * GIB, 'min_available_commit_bytes': 4 * GIB,
                  'min_free_vram_bytes': GIB, 'vram_workspace_bytes': GIB,
                  'max_vram_usage_fraction': .97}
        with patch('moe_prune.memory_safety._available_memory', return_value=(32 * GIB, 32 * GIB)), \
             patch('moe_prune.memory_safety.AdapterMemory') as adapter:
            adapter.return_value.sample.return_value = (64 * GIB, 63 * GIB)
            guard = MemoryGuard(config, {'n_gpu_layers': 1})
            guard.gpu = adapter.return_value
            with self.assertRaises(MemoryError):
                guard.check()


if __name__ == '__main__':
    unittest.main()
