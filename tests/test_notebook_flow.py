"""네트워크·실모델 없이 노트북의 실제 셀과 전체 실험 연결을 실행한다."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from moe_prune.research import canonical_hash, write_json, write_jsonl
from moe_prune.gguf import sha256
from test_gguf_integration import fixture


class SyntheticNative:
    pid = 99999999
    returncode = 0

    def __init__(self, command, **kwargs):
        request = json.loads(Path(command[command.index("--request") + 1]).read_text(encoding="utf-8"))
        output = Path(command[command.index("--output") + 1])
        with output.open("w", encoding="utf-8") as stream:
            if request.get("instrumentation"):
                for phase in ("prefill", "decode"):
                    stream.write(json.dumps({"type": "routing", "phase": phase, "layer": 0, "token_index": 0,
                                             "expert_ids": list(range(8)), "weights": [0.125] * 8,
                                             "weighted_output_norms": list(range(1, 9))}) + "\n")
            stream.write(json.dumps({"type": "result", "text": "42", "token_ids": [42],
                                     "logits": [1.0, 2.0, 3.0], "reference_nll": 0.3,
                                     "reference_tokens": 1, "stop_reason": "eog",
                                     "runtime_commit": "synthetic-not-a-real-inference",
                                     "gpu_memory_bytes": None, "cpu_gpu_placement": None,
                                     "expert_transfer_bytes": None,
                                     "timings": {"load_s": 0.01, "ttft_s": 0.02,
                                                 "prefill_tokens_s": 100.0, "decode_tokens_s": 50.0}}) + "\n")

    def poll(self):
        return 0

    def wait(self):
        return self.returncode


class NotebookFlowTests(unittest.TestCase):
    def test_all_seven_sections_with_synthetic_inputs(self):
        repository = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(repository / "ipynb/01_Gemma4_Expert_Pruning_연구.ipynb", as_version=4)
        headings = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown" and c.source.startswith("## ")]
        self.assertEqual(len(headings), 7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "original.gguf"
            fixture(model, experts=128, top_k=8)
            engine = root / "synthetic.exe"
            engine.write_bytes(b"not executable; Popen is replaced by a test double")
            data = root / "data"
            groups = {}
            for split in ("calibration", "selection", "evaluation"):
                groups[split] = [{"id": f"{split}:{domain}:{i}", "domain": domain, "dataset": f"synthetic-{domain}",
                                  "prompt": f"{split} {domain} unique question {i}",
                                  "reference": "#### 42" if domain == "math" else "return 42" if domain == "code" else ""}
                                 for domain in ("math", "code", "general") for i in range(4)]
                write_jsonl(data / f"{split}.jsonl", groups[split])
            write_json(data / "manifest.json", {
                "file_sha256": {key: sha256(data / f"{key}.jsonl") for key in groups},
                "full_record_hashes": {key: canonical_hash(value) for key, value in groups.items()},
            })
            namespace = {"__name__": "__notebook_synthetic_test__"}
            with patch("moe_prune.research.source_revision", return_value={"git_commit": "synthetic", "git_dirty": False}), \
                 patch("moe_prune.research.baseline_environment", return_value={"gpu": None, "ram_bytes": None}), \
                 patch("moe_prune.research.subprocess.Popen", side_effect=SyntheticNative), \
                 patch("moe_prune.memory_safety._available_memory", return_value=(64 * 1024**3, 64 * 1024**3)), \
                 patch("moe_prune.memory_safety.AdapterMemory") as adapter, \
                 patch("psutil.Process"), contextlib.redirect_stdout(io.StringIO()):
                adapter.return_value.sample.return_value = (16 * 1024**3, 0)
                for keep, previous in ((120, None), (96, 120)):
                    configured = False
                    for index, cell in enumerate(notebook.cells):
                        if cell.cell_type != "code":
                            continue
                        exec(compile(cell.source, f"notebook-cell-{index}", "exec"), namespace)
                        if "INSTALL_DEPENDENCIES = False" in cell.source:
                            configured = True
                            namespace.update(ROOT=root, MODEL=model, ENGINE=engine, DATA=data,
                                             RUN_MODEL=True, RUN_PRUNING_ANALYSIS=True, WRITE_MODELS=True, RUN_EVALUATION=True, RUN_BENCHMARK=True,
                                             APPROVED_KEEP_COUNTS=[keep])
                            namespace["CONFIG"]["engine"] = str(engine)
                            if previous:
                                review = root / "runs" / namespace["RUN_ID"] / f"selection-review-{previous}.json"
                                namespace["PRIOR_SELECTION_REVIEW_HASH"] = sha256(review)
                    self.assertTrue(configured)
                    self.assertEqual(set(namespace["exports"]), {f"frequency-{keep}", f"impact-{keep}"})
            self.assertEqual(set(namespace["evaluations"]), {"original", "frequency-96", "impact-96"})
            self.assertTrue((namespace["exp"].root / "selection-review-120.json").is_file())
            self.assertTrue(namespace["speed_results"])
            self.assertEqual(model.read_bytes()[:4], b"GGUF")


if __name__ == "__main__":
    unittest.main()
