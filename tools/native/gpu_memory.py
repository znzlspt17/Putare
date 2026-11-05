"""2026-11-05: OS 프로세스 dedicated GPU memory 표본. HIP allocation peak가 아니다.

PDH English counter를 localized path로 바꾸고 wildcard를 확장하는 Microsoft 절차 사용:
https://learn.microsoft.com/windows/win32/api/pdh/nf-pdh-pdhaddenglishcounterw
Windows GPU Process Memory counter 자체의 알려진 제한:
https://learn.microsoft.com/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value
"""
from __future__ import annotations
import ctypes as ct
import os

DWORD, STATUS, HANDLE = ct.c_uint32, ct.c_int32, ct.c_void_p
MORE_DATA = 0x800007D2
FMT_LARGE, FMT_NOSCALE = 0x400, 0x1000


class _CounterInfo(ct.Structure):
    _fields_ = [("length", DWORD), ("type", DWORD), ("version", DWORD), ("status", DWORD),
                ("scale", ct.c_int32), ("default_scale", ct.c_int32),
                ("user_data", ct.c_size_t), ("query_user_data", ct.c_size_t), ("full_path", ct.c_wchar_p)]


class _ValueUnion(ct.Union):
    _fields_ = [("large", ct.c_int64), ("double", ct.c_double), ("long", ct.c_int32), ("pointer", ct.c_void_p)]


class _Value(ct.Structure):
    _fields_ = [("status", DWORD), ("value", _ValueUnion)]


def _code(value):
    return int(value) & 0xFFFFFFFF


def _check(value, action):
    if _code(value):
        raise OSError(f"{action}: PDH 0x{_code(value):08X}")


class _PdhCounters:
    def __init__(self, pid):
        if os.name != "nt":
            raise OSError("Windows PDH is unavailable on this OS")
        self.dll = ct.WinDLL("pdh.dll")
        signatures = {
            "PdhOpenQueryW": ([ct.c_wchar_p, ct.c_size_t, ct.POINTER(HANDLE)], STATUS),
            "PdhAddEnglishCounterW": ([HANDLE, ct.c_wchar_p, ct.c_size_t, ct.POINTER(HANDLE)], STATUS),
            "PdhAddCounterW": ([HANDLE, ct.c_wchar_p, ct.c_size_t, ct.POINTER(HANDLE)], STATUS),
            "PdhGetCounterInfoW": ([HANDLE, ct.c_int, ct.POINTER(DWORD), ct.c_void_p], STATUS),
            "PdhExpandWildCardPathW": ([ct.c_wchar_p, ct.c_wchar_p, ct.c_wchar_p, ct.POINTER(DWORD), DWORD], STATUS),
            "PdhCollectQueryData": ([HANDLE], STATUS),
            "PdhGetFormattedCounterValue": ([HANDLE, DWORD, ct.POINTER(DWORD), ct.POINTER(_Value)], STATUS),
            "PdhRemoveCounter": ([HANDLE], STATUS), "PdhCloseQuery": ([HANDLE], STATUS),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = args, result
        self.query, self.counters = HANDLE(), {}
        _check(self.dll.PdhOpenQueryW(None, 0, ct.byref(self.query)), "open query")
        seed = HANDLE()
        try:
            # 정확한 PID 접두사만 요청한다. 다른 프로세스의 counter를 합산하지 않는다.
            english = rf"\GPU Process Memory(pid_{int(pid)}_*)\Dedicated Usage"
            _check(self.dll.PdhAddEnglishCounterW(self.query, english, 0, ct.byref(seed)), "add English counter")
            size = DWORD(0)
            code = self.dll.PdhGetCounterInfoW(seed, 0, ct.byref(size), None)
            if _code(code) != MORE_DATA:
                _check(code, "counter info size")
            if not size.value:
                raise OSError("PDH counter info is empty")
            buffer = ct.create_string_buffer(size.value)
            _check(self.dll.PdhGetCounterInfoW(seed, 0, ct.byref(size), buffer), "counter info")
            self.localized = ct.cast(buffer, ct.POINTER(_CounterInfo)).contents.full_path
            if not self.localized:
                raise OSError("PDH localized path is empty")
        except Exception:
            self.close()
            raise
        finally:
            if seed and self.query:
                self.dll.PdhRemoveCounter(seed)

    def read(self):
        size = DWORD(0)
        code = self.dll.PdhExpandWildCardPathW(None, self.localized, None, ct.byref(size), 0)
        if _code(code) != MORE_DATA:
            _check(code, "expand counter size")
        if not size.value:
            return []
        paths = ct.create_unicode_buffer(size.value)
        _check(self.dll.PdhExpandWildCardPathW(None, self.localized, paths, ct.byref(size), 0), "expand counters")
        current = set("".join(paths[:size.value]).strip("\0").split("\0"))
        current.discard("")
        for path in set(self.counters) - current:
            self.dll.PdhRemoveCounter(self.counters.pop(path))
        for path in current - set(self.counters):
            counter = HANDLE()
            _check(self.dll.PdhAddCounterW(self.query, path, 0, ct.byref(counter)), "add localized counter")
            self.counters[path] = counter
        if not self.counters:
            return []
        _check(self.dll.PdhCollectQueryData(self.query), "collect")
        values = []
        for counter in self.counters.values():
            value = _Value()
            code = self.dll.PdhGetFormattedCounterValue(counter, FMT_LARGE | FMT_NOSCALE, None, ct.byref(value))
            if _code(code) == 0:
                values.append((value.status, value.value.large))
            else:
                # 일부 adapter만 성공한 합계는 실제 프로세스 값으로 보고하지 않는다.
                values.append((_code(code), 0))
        return values

    def close(self):
        if getattr(self, "query", None):
            self.dll.PdhCloseQuery(self.query)
            self.query = None


class ProcessGpuMemorySampler:
    """외부 monitor loop에서 sample() 호출. PID의 여러 adapter/segment counter를 합산한다."""
    def __init__(self, pid, api=None):
        self.pid, self.api = int(pid), api
        self.peak, self.samples, self.reason = None, 0, None
        try:
            if self.pid <= 0:
                raise ValueError("positive PID required")
            if self.api is None:
                self.api = _PdhCounters(self.pid)
        except (OSError, ValueError, AttributeError) as error:
            self.reason = str(error)

    def sample(self):
        if self.api is None:
            return None
        try:
            rows = self.api.read()
            if not rows or any(status not in (0, 1) or not isinstance(value, int) or value < 0 for status, value in rows):
                self.reason = "No complete valid process GPU counter sample"
                return None
            current = sum(value for _, value in rows)
            self.samples += 1
            self.peak = current if self.peak is None else max(self.peak, current)
            self.reason = None
            return current
        except (OSError, ValueError, AttributeError) as error:
            self.reason = str(error)
            return None

    def snapshot(self):
        return {"metric": "windows_process_gpu_dedicated_sampled", "pid": self.pid,
                "peak_bytes": self.peak, "samples": self.samples, "unavailable_reason": self.reason,
                "limitations": "OS PDH process dedicated memory; sums adapters, sampled peak only; not HIP allocation peak, expert-only bytes, or guaranteed physical residency. Windows counters may be inaccurate."}

    def close(self):
        if self.api is not None:
            self.api.close()
            self.api = None
