"""GGUF v3 little-endian 전용, 양자화 블록을 역양자화 없이 보존한다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import os
import re
import shutil
import struct


# 2026-11-05: ggml type별 (block elements, bytes); 알 수 없는 형식은 추정하지 않는다.
STORAGE = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
           7: (32, 24), 8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110),
           12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
           16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
           20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
           24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
           29: (256, 56), 30: (1, 2)}
# IQ4_NL is type 20: 2-byte fp16 scale + sixteen packed values = 18 bytes.
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_allowlist(allowlist, layers, experts, top_k):
    normalized = {int(layer): list(ids) for layer, ids in allowlist.items()}
    if len(normalized) != len(allowlist) or set(normalized) != set(layers):
        raise ValueError("모든 MoE 계층을 정확히 한 번 지정해야 합니다")
    sizes = set()
    for layer, ids in normalized.items():
        if any(type(x) is not int or x < 0 or x >= experts for x in ids):
            raise ValueError(f"범위 밖 Expert ID: layer {layer}")
        if len(set(ids)) != len(ids) or len(ids) < top_k:
            raise ValueError("중복 ID 또는 Top-K 미만의 집합")
        normalized[layer] = sorted(ids)
        sizes.add(len(ids))
    if len(sizes) != 1:
        raise ValueError("현재 런타임은 계층별 동일 유지 개수만 지원합니다")
    return normalized


def _read(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise ValueError("잘린 GGUF")
    return data


def _unpack(stream, fmt):
    return struct.unpack("<" + fmt, _read(stream, struct.calcsize("<" + fmt)))[0]


def _string(stream):
    n = _unpack(stream, "Q")
    if n > 128 * 1024 * 1024:
        raise ValueError("비정상 GGUF 문자열 크기")
    return _read(stream, n).decode("utf-8")


def _value(stream, kind, depth=0):
    if depth > 2:
        raise ValueError("중첩 GGUF 배열 미지원")
    if kind in SCALARS:
        return _unpack(stream, SCALARS[kind])
    if kind == 8:
        return _string(stream)
    if kind == 9:
        subtype, n = _unpack(stream, "I"), _unpack(stream, "Q")
        if n > 10_000_000:
            raise ValueError("비정상 GGUF 배열 크기")
        return [_value(stream, subtype, depth + 1) for _ in range(n)]
    raise ValueError(f"지원하지 않는 metadata type {kind}")


def _pack_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]  # ggml order, first dimension contiguous
    kind: int
    offset: int
    nbytes: int


class GGUF:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = {}
        self.metadata_raw = []
        with self.path.open("rb") as stream:
            if _read(stream, 4) != b"GGUF" or _unpack(stream, "I") != 3:
                raise ValueError("little-endian GGUF v3만 지원합니다")
            nt, nk = _unpack(stream, "Q"), _unpack(stream, "Q")
            if nt > 100_000 or nk > 100_000:
                raise ValueError("비정상 GGUF 항목 수")
            for _ in range(nk):
                start = stream.tell()
                key, kind = _string(stream), _unpack(stream, "I")
                if key in self.metadata:
                    raise ValueError("중복 metadata key")
                self.metadata[key] = _value(stream, kind)
                end = stream.tell()
                stream.seek(start)
                self.metadata_raw.append((key, kind, _read(stream, end - start)))
            self.tensors = []
            for _ in range(nt):
                name, nd = _string(stream), _unpack(stream, "I")
                if not 1 <= nd <= 4:
                    raise ValueError("지원하지 않는 tensor 차원")
                shape = tuple(_unpack(stream, "Q") for _ in range(nd))
                kind, offset = _unpack(stream, "I"), _unpack(stream, "Q")
                if kind not in STORAGE or any(x == 0 for x in shape):
                    raise ValueError(f"지원하지 않는 tensor type/shape: {name}, {kind}")
                block, size = STORAGE[kind]
                if shape[0] % block:
                    raise ValueError(f"양자화 블록 정렬 오류: {name}")
                self.tensors.append(Tensor(name, shape, kind, offset, math.prod(shape) // block * size))
            self.alignment = int(self.metadata.get("general.alignment", 32))
            if self.alignment < 1 or self.alignment > 65536 or self.alignment & (self.alignment - 1):
                raise ValueError("지원하지 않는 alignment")
            self.data_offset = (stream.tell() + self.alignment - 1) // self.alignment * self.alignment
        if len({t.name for t in self.tensors}) != len(self.tensors):
            raise ValueError("중복 tensor 이름")
        prior_end = 0
        for t in sorted(self.tensors, key=lambda t: t.offset):
            if t.offset % self.alignment or t.offset < prior_end or self.data_offset + t.offset + t.nbytes > self.path.stat().st_size:
                raise ValueError(f"tensor 범위/정렬 오류: {t.name}")
            prior_end = t.offset + t.nbytes

    def inspect(self, hash_file=True):
        groups = {k: 0 for k in ("attention", "experts", "router", "other")}
        tensors = []
        for t in self.tensors:
            group = "experts" if "_exps" in t.name else "router" if "ffn_gate_inp" in t.name else "attention" if ".attn_" in t.name else "other"
            groups[group] += t.nbytes
            tensors.append({"name": t.name, "shape": t.shape, "ggml_type": t.kind, "bytes": t.nbytes, "group": group})
        return {"path": str(self.path.resolve()), "sha256": sha256(self.path) if hash_file else None,
                "file_bytes": self.path.stat().st_size, "metadata": self.metadata, "tensor_groups": groups, "tensors": tensors}

    def architecture(self):
        arch = self.metadata.get("general.architecture")
        if arch != "gemma4":
            raise ValueError(f"gemma4 전용 도구입니다: {arch}")
        layers = sorted(int(m.group(1)) for t in self.tensors if (m := re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", t.name)))
        if not layers:
            raise ValueError("MoE router tensor가 없습니다")
        experts = int(self.metadata["gemma4.expert_count"])
        top_k = int(self.metadata["gemma4.expert_used_count"])
        return layers, experts, top_k

    def rewrite(self, destination, allowlist=None, *, permutation=None):
        """2026-11-08: Expert 최외곽 축을 블록 그대로 절삭하거나 전체 순열로 재배치한다."""
        destination = Path(destination)
        if destination.resolve() == self.path.resolve() or destination.exists():
            raise FileExistsError("원본 또는 기존 산출물을 덮어쓸 수 없습니다")
        layers, experts, top_k = self.architecture()
        if allowlist is not None and permutation is not None:
            raise ValueError("절삭 allowlist와 permutation을 동시에 지정할 수 없습니다")
        if permutation is None:
            mask = validate_allowlist({l: list(range(experts)) for l in layers} if allowlist is None else allowlist, layers, experts, top_k)
        else:
            # 2026-11-08: 절삭 정렬 규칙과 분리해 전체 Expert의 명시된 순서를 보존한다.
            checked = validate_allowlist(permutation, layers, experts, top_k)
            if any(len(ids) != experts for ids in checked.values()):
                raise ValueError("순열은 모든 Expert를 정확히 한 번 포함해야 합니다")
            mask = {int(layer): list(ids) for layer, ids in permutation.items()}
        keep = len(next(iter(mask.values())))
        plans = []
        seen = {layer: set() for layer in layers}
        for t in self.tensors:
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", t.name)
            ids = None
            shape = t.shape
            if permutation is not None and ("_exps" in t.name or "ffn_gate_inp." in t.name) and (not match or int(match[1]) not in mask):
                raise ValueError(f"알 수 없는 Expert tensor 계층: {t.name}")
            if match and int(match[1]) in mask:
                layer, suffix = int(match[1]), match[2]
                is_router = suffix == "ffn_gate_inp.weight"
                is_exp = re.fullmatch(r"ffn_(gate|up|down|gate_up)_exps\.(weight|scale)", suffix)
                if is_router or is_exp:
                    if t.shape[-1] != experts:
                        raise ValueError(f"Expert 최외곽 축이 아닙니다: {t.name} {t.shape}")
                    if len(t.shape) == 1 and t.kind not in (0, 1):
                        raise ValueError("양자화된 per-expert scalar 절삭 미지원")
                    ids = mask[layer]
                    shape = t.shape[:-1] + (keep,)
                    seen[layer].add(suffix)
                elif "exps" in suffix or (suffix.startswith("ffn_gate_inp.") and suffix != "ffn_gate_inp.scale"):
                    raise ValueError(f"알 수 없는 Expert tensor: {t.name}")
            size = t.nbytes if ids is None else t.nbytes // experts * keep
            plans.append((t, shape, ids, size))
        for layer, names in seen.items():
            if "ffn_gate_inp.weight" not in names or "ffn_down_exps.weight" not in names or not ({"ffn_gate_exps.weight", "ffn_up_exps.weight"} <= names or "ffn_gate_up_exps.weight" in names):
                raise ValueError(f"필수 Expert tensor 누락: layer {layer}")
        meta = b""
        for key, kind, raw in self.metadata_raw:
            if key == "gemma4.expert_count":
                if kind not in SCALARS or kind in (6, 7, 12):
                    raise ValueError("Expert count 정수 metadata 필요")
                raw = _pack_string(key) + struct.pack("<I", kind) + struct.pack("<" + SCALARS[kind], keep)
            meta += raw
        offset, infos = 0, b""
        for t, shape, _, size in plans:
            offset = (offset + self.alignment - 1) // self.alignment * self.alignment
            infos += _pack_string(t.name) + struct.pack("<I", len(shape)) + struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", t.kind, offset)
            offset += size
        header = b"GGUF" + struct.pack("<IQQ", 3, len(plans), len(self.metadata_raw)) + meta + infos
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_size = len(header) + self.alignment + offset
        if shutil.disk_usage(destination.parent).free < expected_size + 64 * 1024 * 1024:
            raise OSError("GGUF 산출물과 여유 공간을 위한 디스크 용량이 부족합니다")
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            with self.path.open("rb") as src, temporary.open("xb") as out:
                out.write(header)
                out.write(b"\0" * (-out.tell() % self.alignment))
                for t, shape, ids, size in plans:
                    out.write(b"\0" * (-out.tell() % self.alignment))
                    unit = t.nbytes if ids is None else t.nbytes // experts
                    for index in ([0] if ids is None else ids):
                        src.seek(self.data_offset + t.offset + index * unit)
                        remaining = unit
                        while remaining:
                            chunk = _read(src, min(4 * 1024 * 1024, remaining))
                            out.write(chunk)
                            remaining -= len(chunk)
                out.flush()
                os.fsync(out.fileno())
            GGUF(temporary)
            temporary.rename(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return {"path": str(destination.resolve()), "source_sha256": sha256(self.path), "sha256": sha256(destination),
                "file_bytes": destination.stat().st_size, "allowlist": mask,
                "old_to_new": {str(l): {str(old): new for new, old in enumerate(ids)} for l, ids in mask.items()}}
