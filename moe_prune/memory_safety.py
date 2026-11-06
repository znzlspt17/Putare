"""2026-11-05: 실행 가능한 시간 대신 메모리 여유를 감시하는 보수적 보호 장치.

OS GPU 카운터는 표본이며 예약이나 hard allocation limit이 아니다.
https://learn.microsoft.com/en-us/windows/win32/api/dxgi/ns-dxgi-dxgi_adapter_desc1
https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex
"""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import re
import uuid

from tools.native.gpu_memory import _PdhCounters

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


class _MemoryStatus(ct.Structure):
    _fields_ = [('length', ct.c_uint32), ('load', ct.c_uint32),
                ('total_physical', ct.c_uint64), ('available_physical', ct.c_uint64),
                ('total_commit', ct.c_uint64), ('available_commit', ct.c_uint64),
                ('total_virtual', ct.c_uint64), ('available_virtual', ct.c_uint64),
                ('available_extended_virtual', ct.c_uint64)]


def _available_memory():
    if os.name != 'nt':
        raise RuntimeError('Windows commit 메모리 관측을 사용할 수 없습니다')
    status = _MemoryStatus()
    status.length = ct.sizeof(status)
    fn = ct.windll.kernel32.GlobalMemoryStatusEx
    fn.argtypes = [ct.POINTER(_MemoryStatus)]
    fn.restype = ct.c_int
    if not fn(ct.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed')
    return int(status.available_physical), int(status.available_commit)


class _Luid(ct.Structure):
    _fields_ = [('low', ct.c_uint32), ('high', ct.c_int32)]


class _AdapterDesc(ct.Structure):
    _fields_ = [('description', ct.c_wchar * 128), ('vendor', ct.c_uint32),
                ('device', ct.c_uint32), ('subsystem', ct.c_uint32), ('revision', ct.c_uint32),
                ('dedicated_video', ct.c_size_t), ('dedicated_system', ct.c_size_t),
                ('shared_system', ct.c_size_t), ('luid', _Luid), ('flags', ct.c_uint32)]


def _method(obj, index, result, *args):
    table = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(result, ct.c_void_p, *args)(table[index])


def _amd_adapter():
    """DXGI LUID와 PDH adapter 식별자를 결합한다; 선택 GPU가 모호하면 중단한다."""
    if os.name != 'nt':
        raise RuntimeError('Windows DXGI GPU 관측을 사용할 수 없습니다')
    iid = (ct.c_ubyte * 16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    factory = ct.c_void_p()
    create = ct.windll.dxgi.CreateDXGIFactory1
    create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_long
    if create(ct.byref(iid), ct.byref(factory)) < 0:
        raise OSError('CreateDXGIFactory1 failed')
    adapters = []
    try:
        enum = _method(factory, 12, ct.c_long, ct.c_uint32, ct.POINTER(ct.c_void_p))
        index = 0
        while True:
            adapter = ct.c_void_p()
            hr = enum(factory, index, ct.byref(adapter))
            if hr & 0xFFFFFFFF == 0x887A0002:
                break
            if hr < 0 or not adapter.value:
                raise OSError(f'EnumAdapters1 failed: 0x{hr & 0xFFFFFFFF:08x}')
            try:
                desc = _AdapterDesc()
                get_desc = _method(adapter, 10, ct.c_long, ct.POINTER(_AdapterDesc))
                if get_desc(adapter, ct.byref(desc)) < 0:
                    raise OSError('GetDesc1 failed')
                if desc.vendor == 0x1002 and not (desc.flags & 2) and desc.dedicated_video:
                    adapters.append({'name': desc.description, 'capacity': int(desc.dedicated_video),
                                     'luid': (desc.luid.high & 0xFFFFFFFF, desc.luid.low)})
            finally:
                _method(adapter, 2, ct.c_ulong)(adapter)
            index += 1
    finally:
        _method(factory, 2, ct.c_ulong)(factory)
    if len(adapters) != 1:
        raise RuntimeError(f'AMD GPU를 하나로 식별할 수 없습니다: {len(adapters)}개')
    return adapters[0]


class AdapterMemory:
    """전체 adapter dedicated 사용량; PDH 누락을 0으로 간주하지 않는다."""
    def __init__(self):
        self.adapter = _amd_adapter()
        self.api = _PdhCounters(english_path=r'\GPU Adapter Memory(*)\Dedicated Usage')

    def sample(self):
        rows = self.api.read_named()
        high, low = self.adapter['luid']
        pattern = re.compile(rf'luid_0x{high:08x}_0x{low:08x}_phys_\d+', re.I)
        matching = [value for path, value in rows.items() if pattern.search(path)]
        if len(matching) != 1 or matching[0][0] not in (0, 1) or matching[0][1] < 0:
            raise RuntimeError('대상 GPU의 dedicated usage counter를 관측할 수 없습니다')
        return self.adapter['capacity'], int(matching[0][1])

    def close(self):
        self.api.close()


class MemoryGuard:
    def __init__(self, config, request):
        self.config, self.request = config, request
        self.gpu = None
        self.last = {}
        self.ram_reserve = config.get('min_available_ram_bytes', 4 * GIB)
        self.commit_reserve = config.get('min_available_commit_bytes', 4 * GIB)
        self.vram_reserve = config.get('min_free_vram_bytes', GIB)
        self.workspace = config.get('vram_workspace_bytes', GIB)
        self.vram_fraction = config.get('max_vram_usage_fraction', 1.0)
        if isinstance(self.vram_fraction, bool) or not isinstance(self.vram_fraction, (float, int)) or not 0 < self.vram_fraction <= 1:
            raise ValueError('VRAM 사용 비율은 0보다 크고 1 이하이어야 합니다')
        for value in (self.ram_reserve, self.commit_reserve, self.vram_reserve, self.workspace):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError('메모리 여유 값은 양의 정수여야 합니다')

    def preflight(self):
        if self.request.get('n_gpu_layers', 0) > 0:
            self.gpu = AdapterMemory()
        self.check()
        gpu_kv, cpu_kv = (0, 0)
        if 'cache_type_k' in self.request:
            gpu_kv, cpu_kv = _kv_estimates(self.request['model'], self.request)
            self.last.update(vram_estimated_kv_bytes=gpu_kv, ram_estimated_kv_bytes=cpu_kv)
            if self.last['ram_available_bytes'] < self.ram_reserve + cpu_kv:
                raise MemoryError('KV cache를 포함한 물리 RAM 여유 부족')
            if self.last['commit_available_bytes'] < self.commit_reserve + cpu_kv:
                raise MemoryError('KV cache를 포함한 commit 여유 부족')
        if self.gpu:
            weight_bytes = _gpu_weight_estimate(self.request['model'], self.request['n_gpu_layers'])
            kv_bytes = gpu_kv
            fraction_reserve = self.last['vram_capacity_bytes'] - int(self.last['vram_capacity_bytes'] * self.vram_fraction)
            required = weight_bytes + kv_bytes + self.workspace + max(self.vram_reserve, fraction_reserve)
            self.last.update(vram_estimated_model_bytes=weight_bytes, vram_estimated_kv_bytes=kv_bytes,
                             vram_preflight_required_bytes=required)
            if self.last['vram_free_bytes'] < required:
                raise MemoryError(f'GPU reserve 부족: free={self.last["vram_free_bytes"]}, 예상 필요={required}')
        return dict(self.last)

    def check(self):
        physical, commit = _available_memory()
        self.last = {'ram_available_bytes': physical, 'commit_available_bytes': commit}
        if physical < self.ram_reserve:
            raise MemoryError(f'물리 RAM 여유 부족: {physical} < {self.ram_reserve}')
        if commit < self.commit_reserve:
            raise MemoryError(f'commit reserve 부족: {commit} < {self.commit_reserve}')
        if self.gpu:
            total, used = self.gpu.sample()
            if total <= 0 or used > total:
                raise RuntimeError('GPU memory counter가 물리 VRAM과 일치하지 않습니다')
            free = total - used
            self.last.update(vram_capacity_bytes=total, vram_used_bytes=used, vram_free_bytes=free)
            if free < self.vram_reserve:
                raise MemoryError(f'GPU reserve 부족: {free} < {self.vram_reserve}')
            if used > int(total * self.vram_fraction):
                raise MemoryError(f'GPU 사용 비율 초과: {used} / {total}')
        return dict(self.last)

    def close(self):
        if self.gpu:
            self.gpu.close()
            self.gpu = None
