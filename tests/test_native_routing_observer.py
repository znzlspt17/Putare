"""고정 llama.cpp header와 합성 텐서로 실제 C++ callback을 검증한다."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/llama.cpp'


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++') and (UPSTREAM / 'include/llama.h').is_file(),
                     'Linux g++와 고정 llama.cpp 소스가 필요합니다')
class RoutingObserverNativeTests(unittest.TestCase):
    def test_lightweight_and_legacy_observer_on_synthetic_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / 'observer-test'
            output = Path(folder) / 'events.jsonl'
            command = [shutil.which('g++'), '-std=c++17', '-O0', '-ffunction-sections', '-fdata-sections',
                       '-Wl,--gc-sections', '-DRESEARCH_LLAMA_COMMIT=1',
                       *[f'-I{UPSTREAM / path}' for path in ('include', 'ggml/include', 'common', 'vendor')],
                       str(ROOT / 'tools/native/test_routing_observer.cpp'), '-o', str(exe)]
            subprocess.run(command, check=True, timeout=120)
            subprocess.run([str(exe), str(output)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
