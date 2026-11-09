"""Linux 전체 RAM과 NVML GPU 메모리의 경고 기준을 감시한다."""
from __future__ import annotations

from decimal import Decimal
import os
from pathlib import Path
import re

from tools.native.gpu_memory import NvmlDevice

GIB = 1024 ** 3


def _gpu_weight_estimate(model_path, n_gpu_layers):
    """2026-11-06: pinned llama.cpp 배치를 따른 보수적 weight 추정이며 메모리 예약은 아니다."""
    from .gguf import GGUF

    if type(n_gpu_layers) is not int or n_gpu_layers <= 0:
        raise ValueError('GPU layer 수는 양의 정수여야 합니다')
    model = GGUF(model_path)
    count = model.metadata.get('gemma4.block_count')
    # pinned llama.cpp의 LLAMA_MAX_LAYERS와 같은 한계로 비정상 metadata의 과도한 할당을 막는다.
    if model.metadata.get('general.architecture') != 'gemma4' or type(count) is not int or not 1 <= count <= 512:
        raise ValueError('Gemma4 block_count를 확인할 수 없습니다')
    # 고정 revision은 input을 CPU에 두고 마지막 repeating layers와 output을 GPU에 배정한다.
    # 비반복 tensor 전부를 GPU에 있다고 가정해 tied input/output 등은 축소하지 않는다.
    start = max(count + 1 - n_gpu_layers, 0)
    blocks, seen, nonblock = [], set(), []
    for tensor in model.tensors:
        match = re.fullmatch(r'blk\.(\d+)\..+', tensor.name)
        if tensor.name.startswith('blk.'):
            if not match or int(match.group(1)) >= count:
                raise ValueError(f'지원하지 않는 GPU layer tensor: {tensor.name}')
            layer = int(match.group(1))
            seen.add(layer)
            if layer >= start:
                blocks.append(tensor)
        else:
            nonblock.append(tensor)
    if seen != set(range(count)) or not any(t.name == 'token_embd.weight' for t in nonblock):
        raise ValueError('Gemma4 layer 또는 token embedding tensor가 누락되었습니다')
    # 256바이트 정렬은 tensor별 계산의 여유일 뿐 backend buffer 크기 보장이 아니다.
    rounded = lambda tensor: (tensor.nbytes + 255) // 256 * 256
    estimate = sum(rounded(t) for t in blocks + nonblock)
    # Gemma4는 전역 rope_freqs를 여러 반복 계층에서 중복 생성할 수 있다.
    rope = sum(rounded(t) for t in nonblock if t.name.startswith('rope_freqs.'))
    estimate += rope * max(min(n_gpu_layers, count) - 1, 0)
    if n_gpu_layers >= count + 1:
        estimate = max(estimate, Path(model_path).stat().st_size)
    return estimate


def _kv_estimates(model_path, request):
    """2026-11-06: Gemma4 SWA·공유 KV와 Q8 블록 크기를 반영하되 예약으로 취급하지 않는다."""
    from .gguf import GGUF
    m = GGUF(model_path).metadata
    count = m['gemma4.block_count']
    heads = m['gemma4.attention.head_count_kv']
    heads = [heads] * count if type(heads) is int else heads
    swa = m.get('gemma4.attention.sliding_window_pattern', [False] * count)
    if len(heads) != count or len(swa) != count:
        raise ValueError('KV head 또는 SWA metadata 형상 불일치')
    ctx = ((int(request['context_size']) + 255) // 256) * 256
    # 2026-11-06: 이 pinned llama.cpp의 llama_context_default_params는 swa_full=true다.
    # 전체 크기를 기준으로 삼아 GPU 메모리를 과소 추정하지 않는다.
    size_swa = ctx
    shared = int(m.get('gemma4.attention.shared_kv_layers', 0))
    start = max(count + 1 - request['n_gpu_layers'], 0)
    gpu_total, cpu_total = 0, 0
    for layer in range(0, count - shared):
        is_swa = bool(swa[layer])
        cells = size_swa if is_swa else ctx
        suffix = '_swa' if is_swa else ''
        for kind, key in (('k', 'key'), ('v', 'value')):
            width = int(m.get(f'gemma4.attention.{key}_length{suffix}', m[f'gemma4.attention.{key}_length'])) * int(heads[layer])
            cache_type = request.get(f'cache_type_{kind}', 'f16')
            if cache_type == 'q8_0':
                if width % 32:
                    raise ValueError('Q8 KV width는 32의 배수여야 합니다')
                row = width // 32 * 34
            elif cache_type == 'f16':
                row = width * 2
            else:
                raise ValueError('지원하지 않는 KV cache 형식')
            size = row * cells
            if request.get('offload_kqv', True) and request.get('n_gpu_layers', 0) > 0 and layer >= start:
                gpu_total += size
            else:
                cpu_total += size
    return gpu_total, cpu_total


def _gpu_kv_estimate(model_path, request):
    return _kv_estimates(model_path, request)[0]


# 2026-11-09: container 한도와 상위 cgroup 한도를 함께 확인해 호스트 용량을 과대평가하지 않는다.
def _cgroup_limits():
    if os.name != 'posix' or not Path('/proc/self/cgroup').exists():
        raise RuntimeError('Linux cgroup 정보를 확인할 수 없습니다')
    memberships = []
    for row in Path('/proc/self/cgroup').read_text(encoding='utf-8').splitlines():
        fields = row.split(':', 2)
        if len(fields) == 3 and (fields[1] == '' or 'memory' in fields[1].split(',')):
            memberships.append((fields[1] == '', fields[2]))
    mounts = []
    for row in Path('/proc/self/mountinfo').read_text(encoding='utf-8').splitlines():
        left, separator, right = row.partition(' - ')
        if not separator:
            continue
        before, after = left.split(), right.split()
        if len(before) < 5 or len(after) < 3:
            continue
        if after[0] == 'cgroup2' or (after[0] == 'cgroup' and 'memory' in after[2].split(',')):
            mounts.append((after[0] == 'cgroup2', before[3].replace('\\040',' '), before[4].replace('\\040',' ')))
    limits = []
    for unified, membership in memberships:
        matched = False
        for is_v2, mount_root, mount_point in mounts:
            if unified != is_v2:
                continue
            root = Path(mount_root)
            try:
                relative = Path(membership).relative_to(root)
            except ValueError:
                # cgroup namespace 안에서는 membership의 /가 mount root를 뜻한다.
                if membership == '/':
                    relative = Path('.')
                else:
                    continue
            matched = True
            mount = Path(mount_point)
            current = mount / relative
            while True:
                limit_file = current / ('memory.max' if unified else 'memory.limit_in_bytes')
                used_file = current / ('memory.current' if unified else 'memory.usage_in_bytes')
                if limit_file.exists():
                    raw = limit_file.read_text(encoding='ascii').strip()
                    if raw != 'max':
                        limit, used = int(raw), int(used_file.read_text(encoding='ascii').strip())
                        if limit <= 0 or used < 0:
                            raise RuntimeError('cgroup 메모리 관측값 오류')
                        limits.append((limit, used))
                if current == mount:
                    break
                current = current.parent
        if not matched:
            raise RuntimeError('메모리 cgroup 경로를 확인할 수 없습니다')
    if memberships and not mounts:
        raise RuntimeError('메모리 cgroup mount를 찾을 수 없습니다')
    return limits


def _ram_sample():
    import psutil
    memory = psutil.virtual_memory()
    total, available = int(memory.total), int(memory.available)
    if total <= 0 or not 0 <= available <= total:
        raise RuntimeError('시스템 RAM 관측값 오류')
    source = 'system'
    for limit, used in _cgroup_limits():
        if limit < total:
            total, source = limit, 'cgroup'
        available = min(available, max(0, limit - used))
    return total, min(total, available), source


class AdapterMemory(NvmlDevice):
    """선택된 CUDA GPU의 전체 메모리 사용량."""


# 2026-11-09: 비율 초과는 중단 조건이 아니며 경계 재진입 때만 새 경고를 기록한다.
class MemoryGuard:
    def __init__(self, config, request):
        self.config, self.request = config, request
        obsolete = {'ram_budget_bytes', 'min_available_ram_bytes', 'min_available_commit_bytes',
                    'min_free_vram_bytes', 'vram_workspace_bytes',
                    'max_ram_usage_fraction', 'max_vram_usage_fraction'}
        if set(config) & obsolete:
            raise ValueError('기존 고정 메모리 설정을 제거하고 경고 비율을 사용하세요')
        self.ram_fraction = config.get('ram_usage_warn_fraction', .80)
        self.vram_fraction = config.get('vram_usage_warn_fraction', .97)
        for value in (self.ram_fraction, self.vram_fraction):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError('메모리 경고 비율은 0보다 크고 1 이하이어야 합니다')
        self.gpu, self.last = None, {}
        self.active = {'ram': False, 'vram': False}
        self.warning_events = []

    def _warning(self, resource, used, total, fraction, *, predicted=False):
        exceeds = used > int(Decimal(total) * Decimal(str(fraction)))
        if predicted:
            if exceeds:
                event = {'resource': resource, 'kind': 'predicted', 'used_bytes': used, 'total_bytes': total, 'threshold': fraction}
                self.warning_events.append(event)
                if not self.active[resource]:
                    print(f'메모리 예상 경고: {resource} {used}/{total} > {fraction:.0%}', flush=True)
            return exceeds
        if exceeds and not self.active[resource]:
            event = {'resource': resource, 'kind': 'observed', 'used_bytes': used, 'total_bytes': total, 'threshold': fraction}
            self.warning_events.append(event)
            print(f'메모리 사용 경고: {resource} {used}/{total} > {fraction:.0%}', flush=True)
        self.active[resource] = exceeds
        return exceeds

    def preflight(self):
        if self.request.get('n_gpu_layers', 0) > 0 or self.request.get('monitor_gpu', False):
            self.gpu = AdapterMemory(self.config.get('gpu_uuid'))
        self.check()
        gpu_kv, cpu_kv = (0, 0)
        if 'cache_type_k' in self.request:
            gpu_kv, cpu_kv = _kv_estimates(self.request['model'], self.request)
            self.last.update(vram_estimated_kv_bytes=gpu_kv, ram_estimated_kv_bytes=cpu_kv)
            self._warning('ram', self.last['ram_used_bytes'] + cpu_kv, self.last['ram_total_bytes'], self.ram_fraction, predicted=True)
        if self.gpu and self.request.get('n_gpu_layers', 0) > 0:
            weights = _gpu_weight_estimate(self.request['model'], self.request['n_gpu_layers'])
            self.last.update(vram_estimated_model_bytes=weights, vram_estimated_kv_bytes=gpu_kv)
            self._warning('vram', self.last['vram_used_bytes'] + weights + gpu_kv, self.last['vram_capacity_bytes'], self.vram_fraction, predicted=True)
        self.last['warning_events'] = list(self.warning_events)
        return dict(self.last)

    def check(self):
        total, available, source = _ram_sample()
        if total <= 0 or not 0 <= available <= total:
            raise RuntimeError('RAM 메모리 관측값 오류')
        used = total - available
        self.last = {'ram_total_bytes': total, 'ram_available_bytes': available, 'ram_used_bytes': used,
                     'ram_source': source, 'ram_usage_fraction': used / total,
                     'ram_usage_warn_fraction': self.ram_fraction,
                     'ram_warning': self._warning('ram', used, total, self.ram_fraction)}
        if self.gpu:
            capacity, gpu_used = self.gpu.sample()
            if capacity <= 0 or not 0 <= gpu_used <= capacity:
                raise RuntimeError('GPU 메모리 관측값 오류')
            self.last.update(vram_capacity_bytes=capacity, vram_used_bytes=gpu_used,
                             vram_free_bytes=capacity-gpu_used, vram_usage_fraction=gpu_used/capacity,
                             vram_usage_warn_fraction=self.vram_fraction,
                             vram_warning=self._warning('vram', gpu_used, capacity, self.vram_fraction))
        self.last['warning_events'] = list(self.warning_events)
        return dict(self.last)

    def close(self):
        if self.gpu:
            self.gpu.close()
            self.gpu = None
