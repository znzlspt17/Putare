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
        for value in (self.ram_reserve, self.commit_reserve, self.vram_reserve, self.workspace):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError('메모리 여유 값은 양의 정수여야 합니다')

    def preflight(self):
        if self.request.get('n_gpu_layers', 0) > 0:
            self.gpu = AdapterMemory()
        self.check()
        if self.gpu:
            required = Path(self.request['model']).stat().st_size + self.workspace + self.vram_reserve
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
        return dict(self.last)

    def close(self):
        if self.gpu:
            self.gpu.close()
            self.gpu = None
