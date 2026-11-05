"""실제 모델 없이 GGUF 양자화 바이트와 Expert 축 보존을 확인한다."""
from pathlib import Path
import struct
import tempfile
import unittest

from moe_prune.gguf import GGUF


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def fixture(path, extra=None, experts=4, top_k=2):
    # 2026-11-05: 역양자화 없이 Expert별 서로 다른 블록을 복사하는지 검증한다.
    metadata = [
        ("general.architecture", 8, string("gemma4")),
        ("gemma4.expert_count", 4, struct.pack("<I", experts)),
        ("gemma4.expert_used_count", 4, struct.pack("<I", top_k)),
        ("tokenizer.chat_template", 8, string("그대로 보존")),
    ]
    tensors = [
        ("blk.0.ffn_gate_inp.weight", (32, experts), 20, b"".join(bytes([i + 1]) * 18 for i in range(experts))),
        ("blk.0.ffn_gate_inp.scale", (32,), 0, struct.pack("<32f", *range(32))),
        ("blk.0.ffn_gate_up_exps.weight", (32, 64, experts), 20, b"".join(bytes([i + 10]) * (18 * 64) for i in range(experts))),
        ("blk.0.ffn_down_exps.weight", (32, 32, experts), 20, b"".join(bytes([i + 20]) * (18 * 32) for i in range(experts))),
        ("blk.0.ffn_down_exps.scale", (experts,), 0, struct.pack("<" + "f" * experts, *((i + 1) / experts for i in range(experts)))),
        ("blk.0.ffn_up.weight", (32, 32), 20, bytes([99]) * (18 * 32)),
        ("blk.0.attn_q.weight", (32, 32), 20, bytes([77]) * (18 * 32)),
    ]
    if extra:
        tensors.append(extra)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, kind, data in metadata:
        header += string(key) + struct.pack("<I", kind) + data
    offset = 0
    for name, shape, kind, data in tensors:
        offset += -offset % 32
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", kind, offset)
        offset += len(data)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(b"\0" * (-stream.tell() % 32))
        for _, _, _, data in tensors:
            stream.write(b"\0" * (-stream.tell() % 32))
            stream.write(data)
    return {name: data for name, _, _, data in tensors}


def payloads(model):
    with model.path.open("rb") as stream:
        result = {}
        for tensor in model.tensors:
            stream.seek(model.data_offset + tensor.offset)
            result[tensor.name] = stream.read(tensor.nbytes)
        return result


class PhysicalPruningTests(unittest.TestCase):
    def test_noop_preserves_all_metadata_and_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.gguf", Path(directory) / "noop.gguf"
            expected = fixture(source)
            original_bytes = source.read_bytes()
            GGUF(source).rewrite(target)
            self.assertEqual(GGUF(source).metadata, GGUF(target).metadata)
            self.assertEqual(payloads(GGUF(target)), expected)
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_pruning_preserves_quantized_blocks_and_distinct_scale_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.gguf", Path(directory) / "pruned.gguf"
            expected = fixture(source)
            result = GGUF(source).rewrite(target, {0: [3, 1]})
            pruned, actual = GGUF(target), payloads(GGUF(target))
            self.assertEqual(pruned.metadata["gemma4.expert_count"], 2)
            self.assertEqual(pruned.metadata["gemma4.expert_used_count"], 2)
            for name in ("blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.ffn_down_exps.scale"):
                unit = len(expected[name]) // 4
                self.assertEqual(actual[name], expected[name][unit:2 * unit] + expected[name][3 * unit:4 * unit])
            for name in ("blk.0.ffn_gate_inp.scale", "blk.0.ffn_up.weight", "blk.0.attn_q.weight"):
                self.assertEqual(actual[name], expected[name])
            self.assertEqual(result["old_to_new"], {"0": {"1": 0, "3": 1}})
            self.assertLess(target.stat().st_size, source.stat().st_size)

    def test_empty_allowlist_is_not_treated_as_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.gguf", Path(directory) / "bad.gguf"
            fixture(source)
            with self.assertRaises(ValueError):
                GGUF(source).rewrite(target, {})
            self.assertFalse(target.exists())

    def test_unknown_router_axis_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.gguf", Path(directory) / "bad.gguf"
            fixture(source, ("blk.0.ffn_gate_inp.bias", (4,), 0, b"\0" * 16))
            with self.assertRaises(ValueError):
                GGUF(source).rewrite(target, {0: [0, 1]})
            self.assertFalse(target.exists())

    def test_existing_outputs_and_original_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.gguf", Path(directory) / "exists.gguf"
            fixture(source)
            target.write_bytes(b"user output")
            model = GGUF(source)
            for destination in (source, target):
                with self.assertRaises(FileExistsError):
                    model.rewrite(destination)
            self.assertEqual(target.read_bytes(), b"user output")


if __name__ == "__main__":
    unittest.main()
