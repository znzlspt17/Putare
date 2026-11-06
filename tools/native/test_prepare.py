"""2026-11-05: 실제 모델·compiler 없이 빌드 계약의 위험한 암묵값을 검증한다."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

spec = importlib.util.spec_from_file_location("native_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_no_gpu_architecture_is_invented(self):
        for value in ("", "9070XT", "gfx1201;remove", "gfx"):
            with self.assertRaises(ValueError):
                prepare.build_commands(Path("source"), Path("build"), value)

    def test_build_is_pinned_and_uses_real_hip_options(self):
        commands = prepare.build_commands(Path("a source"), Path("a build"), "gfx1201")
        self.assertEqual(len(prepare.COMMIT), 40)
        self.assertIn("-DGGML_HIP=ON", commands[0])
        self.assertIn("-DHIP_PLATFORM=amd", commands[0])
        self.assertIn("-DGPU_TARGETS=gfx1201", commands[0])
        self.assertIn("-DLLAMA_SOURCE_DIR=a source", commands[0])
        self.assertEqual(commands[1][-3:], ["llama-expert-research", "--parallel", "2"])

    def test_probe_does_not_claim_missing_tool_available(self):
        tools = prepare.preflight()
        self.assertIn("clang++", tools)
        self.assertTrue(all(value is None or isinstance(value, str) for value in tools.values()))

    def test_sdk_device_bitcode_requires_real_directory(self):
        with tempfile.TemporaryDirectory() as root:
            sdk = Path(root)
            self.assertIsNone(prepare.sdk_device_lib_path(sdk))
            bitcode = sdk / 'lib' / 'llvm' / 'amdgcn' / 'bitcode'
            bitcode.mkdir(parents=True)
            self.assertEqual(prepare.sdk_device_lib_path(sdk), bitcode)

    def test_discovers_workspace_local_build_tools_and_current_python_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cmake = root / '.cache/native-tools/cmake/data/bin/cmake.exe'
            ninja = root / '.cache/native-tools/bin/ninja.exe'
            sdk = root / '.venv/Lib/site-packages/_rocm_sdk_devel'
            clang = sdk / 'lib/llvm/bin/clang.exe'
            clangxx = sdk / 'lib/llvm/bin/clang++.exe'
            for path in (cmake, ninja, clang, clangxx):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            found = prepare.discover_local_tools(root, root / '.venv')
            self.assertEqual(found['cmake'], cmake)
            self.assertEqual(found['ninja'], ninja)
            self.assertEqual(found['clang'], clang)
            self.assertEqual(found['clang++'], clangxx)
            self.assertEqual(found['rocm'], sdk)

    def test_runtime_masks_allow_single_layer_ablation(self):
        # 2026-11-05: compiler 부재 시 source 계약만 검증한다. native 실행 검증은 아니다.
        source = Path(__file__).with_name("expert_research.cpp").read_text(encoding="utf-8")
        self.assertNotIn("state.allow[i].size()!=retained", source)
        self.assertIn("state.allow[i].size()<static_cast<size_t>(state.top_k)", source)

    def test_in_process_warmup_clears_kv_before_measurement(self):
        source = Path(__file__).with_name("expert_research.cpp").read_text(encoding="utf-8")
        self.assertIn("llama_memory_clear(memory,true)", source)
        self.assertLess(source.index("llama_memory_clear(memory,true)"), source.index("const auto prefill_start="))
        self.assertIn('warmup requires uninstrumented generate mode', source)


if __name__ == "__main__":
    unittest.main()
