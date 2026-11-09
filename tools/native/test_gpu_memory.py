import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tools.native.gpu_memory import NvmlDevice, ProcessGpuMemorySampler

class FakeNvml:
    NVML_VALUE_NOT_AVAILABLE=0xffffffffffffffff
    def __init__(self,rows=()):self.rows=iter(rows);self.closed=False
    def nvmlInit(self):pass
    def nvmlShutdown(self):self.closed=True
    def nvmlDeviceGetCount(self):return 1
    def nvmlDeviceGetHandleByIndex(self,index):return index
    def nvmlDeviceGetHandleByUUID(self,uuid):return 0
    def nvmlDeviceGetUUID(self,handle):return 'GPU-test'
    def nvmlDeviceGetName(self,handle):return 'NVIDIA A100'
    def nvmlDeviceGetMemoryInfo(self,handle):return SimpleNamespace(total=100,used=30)
    def nvmlDeviceGetComputeRunningProcesses(self,handle):
        row=next(self.rows)
        if isinstance(row,Exception):raise row
        return row

class GpuMemoryTests(unittest.TestCase):
    def sampler(self,rows):
        nvml=FakeNvml(rows)
        api=SimpleNamespace(nvml=nvml,handle=0,close=nvml.nvmlShutdown)
        return ProcessGpuMemorySampler(42,api),nvml
    def test_tracks_only_selected_process_and_peak(self):
        sampler,_=self.sampler([[SimpleNamespace(pid=42,usedGpuMemory=100),SimpleNamespace(pid=9,usedGpuMemory=999)], [SimpleNamespace(pid=42,usedGpuMemory=50)]])
        with patch('psutil.Process',return_value=SimpleNamespace(children=lambda recursive:[])):
            self.assertEqual(sampler.sample(),100);self.assertEqual(sampler.sample(),50)
        self.assertEqual(sampler.snapshot()['peak_bytes'],100);self.assertEqual(sampler.snapshot()['samples'],2)
    def test_missing_and_invalid_process_memory_are_unknown(self):
        sampler,_=self.sampler([[],[SimpleNamespace(pid=42,usedGpuMemory=FakeNvml.NVML_VALUE_NOT_AVAILABLE)]])
        with patch('psutil.Process',return_value=SimpleNamespace(children=lambda recursive:[])):
            self.assertIsNone(sampler.sample());self.assertIsNone(sampler.sample())
        self.assertIsNone(sampler.snapshot()['peak_bytes'])
    def test_observed_zero_is_distinct_from_unknown(self):
        sampler,_=self.sampler([[SimpleNamespace(pid=42,usedGpuMemory=0)]])
        with patch('psutil.Process',return_value=SimpleNamespace(children=lambda recursive:[])):
            self.assertEqual(sampler.sample(),0)
        self.assertEqual(sampler.snapshot()['peak_bytes'],0)
    def test_failure_records_reason_and_closes(self):
        sampler,nvml=self.sampler([OSError('counter')])
        with patch('psutil.Process',return_value=SimpleNamespace(children=lambda recursive:[])):self.assertIsNone(sampler.sample())
        self.assertEqual(sampler.snapshot()['unavailable_reason'],'OSError');sampler.close();self.assertTrue(nvml.closed)
    def test_selected_cuda_device_is_validated(self):
        nvml=FakeNvml()
        with patch.dict(sys.modules,{'pynvml':nvml}), patch('tools.native.gpu_memory.os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
            device=NvmlDevice('GPU-test');self.assertEqual(device.sample(),(100,30));device.close()
            with self.assertRaisesRegex(RuntimeError,'다릅니다'):NvmlDevice('GPU-other')
    def test_blackwell_gpu_is_accepted(self):
        nvml=FakeNvml()
        nvml.nvmlDeviceGetName=lambda handle: 'NVIDIA RTX PRO 6000 Blackwell'
        with patch.dict(sys.modules,{'pynvml':nvml}), patch('tools.native.gpu_memory.os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
            device=NvmlDevice()
            self.assertEqual(device.name, 'NVIDIA RTX PRO 6000 Blackwell')
            device.close()
    def test_ambiguous_cuda_devices_rejected(self):
        nvml=FakeNvml()
        with patch.dict(sys.modules,{'pynvml':nvml}), patch('tools.native.gpu_memory.os.environ',{'CUDA_VISIBLE_DEVICES':'0,1'}):
            with self.assertRaisesRegex(RuntimeError,'단일'):NvmlDevice()
    def test_numeric_device_with_multiple_gpus_is_ambiguous(self):
        nvml=FakeNvml()
        nvml.nvmlDeviceGetCount=lambda: 2
        with patch.dict(sys.modules,{'pynvml':nvml}), patch('tools.native.gpu_memory.os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(RuntimeError,'UUID'):NvmlDevice()

if __name__=='__main__':unittest.main()
