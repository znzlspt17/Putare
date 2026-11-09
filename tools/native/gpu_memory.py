"""Linux NVML GPU 메모리 관측. 표본은 실제 최고 사용량을 보장하지 않는다."""
import os
import psutil


# 2026-11-09: CUDA와 NVML이 서로 다른 장치를 측정하는 상황을 거부한다.
class NvmlDevice:
    def __init__(self, gpu_uuid=None):
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError('NVML Python 바인딩이 필요합니다') from error
        self.nvml = pynvml
        pynvml.nvmlInit()
        try:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible is not None:
                entries = [x.strip() for x in visible.split(',') if x.strip()]
                if len(entries) != 1:
                    raise RuntimeError('단일 CUDA 장치를 지정해야 합니다')
                token = entries[0]
                if token.isdecimal() and pynvml.nvmlDeviceGetCount() != 1:
                    raise RuntimeError('여러 GPU에서는 CUDA_VISIBLE_DEVICES에 UUID를 지정하세요')
                handle = (pynvml.nvmlDeviceGetHandleByIndex(int(token)) if token.isdecimal()
                          else pynvml.nvmlDeviceGetHandleByUUID(token.encode()))
            else:
                if pynvml.nvmlDeviceGetCount() != 1:
                    raise RuntimeError('CUDA 장치가 모호합니다')
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            self.uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
            if gpu_uuid is not None and gpu_uuid != self.uuid:
                raise RuntimeError('요청 GPU와 CUDA_VISIBLE_DEVICES가 다릅니다')
            name = pynvml.nvmlDeviceGetName(handle)
            self.name = name.decode() if isinstance(name, bytes) else str(name)
            # 2026-11-09: NVML이 식별한 단일 GPU를 측정해 A100과 Blackwell을 모두 지원한다.
            self.handle = handle
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def sample(self):
        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        total, used = int(memory.total), int(memory.used)
        if total <= 0 or not 0 <= used <= total:
            raise RuntimeError('GPU 메모리 관측값이 잘못되었습니다')
        return total, used

    def close(self):
        self.nvml.nvmlShutdown()


# 2026-11-09: 프로세스 측정은 전체 GPU 사용률과 구별하며 지원되지 않으면 미측정으로 남긴다.
class ProcessGpuMemorySampler:
    def __init__(self, pid, api=None):
        self.pid, self.peak, self.samples = pid, None, 0
        self.reason = None
        try:
            self.device = api or NvmlDevice()
        except Exception as error:
            self.device = None
            self.reason = type(error).__name__

    def sample(self):
        if self.device is None:
            return
        try:
            device = self.device
            process = psutil.Process(self.pid)
            ids = {self.pid, *(child.pid for child in process.children(recursive=True))}
            nvml = device.nvml
            rows = nvml.nvmlDeviceGetComputeRunningProcesses(device.handle)
            matching = [p for p in rows if p.pid in ids]
            if not matching:
                self.reason = 'ProcessNotObserved'
                return None
            unavailable = getattr(nvml, 'NVML_VALUE_NOT_AVAILABLE', 0xffffffffffffffff)
            if any(p.usedGpuMemory is None or p.usedGpuMemory == unavailable for p in matching):
                raise RuntimeError('프로세스 GPU 메모리 미지원')
            used = sum(int(p.usedGpuMemory) for p in matching)
            if used < 0:
                raise RuntimeError('프로세스 GPU 메모리 값 오류')
            self.peak = max(self.peak or 0, used)
            self.samples += 1
            self.reason = None
            return used
        except Exception as error:
            self.reason = type(error).__name__

    def snapshot(self):
        return {'peak_bytes': self.peak, 'samples': self.samples, 'unavailable_reason': self.reason,
                'limitations': 'NVML 프로세스 메모리 표본 최대치; 물리적 최고치 보장 없음.'}

    def close(self):
        if self.device is not None:
            self.device.close()
            self.device = None
