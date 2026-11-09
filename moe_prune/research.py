"""실험 데이터·통계·재현성 및 사용자 실행 오케스트레이션."""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import platform
import importlib.metadata

from .gguf import GGUF, sha256, validate_allowlist
from .memory_safety import MemoryGuard


_NATIVE_ENV_ALLOWLIST = (
    'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'LD_LIBRARY_PATH',
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
    'GGML_CUDA_NO_PINNED', 'GGML_CUDA_ENABLE_UNIFIED_MEMORY', 'GGML_CUDA_DISABLE_GRAPHS',
)


def spawn_native(command, **kwargs):
    """허용된 Linux/CUDA 변수만 자식에 전달해 비밀 복사를 피한다."""
    environment = {name: value for name in _NATIVE_ENV_ALLOWLIST if (value := os.environ.get(name)) is not None}
    return subprocess.Popen(command, env=environment, **kwargs)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def prompt_hash(prompt):
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def assert_disjoint(groups):
    ids, prompts = {}, {}
    for name, records in groups.items():
        for record in records:
            key = record["id"]
            digest = prompt_hash(record["prompt"])
            if key in ids or digest in prompts:
                raise ValueError(f"중복/누수: {name} {key}")
            ids[key], prompts[digest] = name, name


def split_records(records, calibration, selection, seed=42):
    records = list(records)
    assert_disjoint({"input": records})
    if min(calibration, selection) < 0 or calibration + selection > len(records):
        raise ValueError("데이터 수보다 큰 분할")
    random.Random(seed).shuffle(records)
    return {"calibration": records[:calibration], "selection": records[calibration:calibration + selection], "evaluation": records[calibration + selection:]}


def aggregate_routing(events, experts):
    """2026-11-05: 기여도 미관측 상태를 0과 구분해 저장한다."""
    groups = {}
    for event in events:
        if event.get("type") != "routing":
            continue
        key = (event["domain"], int(event["layer"]), event["phase"])
        if event["phase"] not in ("prefill", "decode"):
            raise ValueError("알 수 없는 추론 단계")
        ids, weights = event["expert_ids"], event["weights"]
        norms = event.get("weighted_output_norms")
        if len(ids) != len(weights) or len(set(ids)) != len(ids) or (norms is not None and len(norms) != len(ids)):
            raise ValueError("Routing event 형상 불일치")
        row = groups.setdefault(key, {"frequency": [0.] * experts, "gate_weight": [0.] * experts, "contribution": [0.] * experts, "contribution_observed": True, "tokens": 0})
        row["tokens"] += 1
        row["contribution_observed"] &= norms is not None
        for i, (expert, weight) in enumerate(zip(ids, weights)):
            if type(expert) is not int or not 0 <= expert < experts or not math.isfinite(weight) or weight < 0:
                raise ValueError("유효하지 않은 Routing 값")
            row["frequency"][expert] += 1
            row["gate_weight"][expert] += weight
            if norms is not None:
                if not math.isfinite(norms[i]) or norms[i] < 0:
                    raise ValueError("유효하지 않은 contribution")
                row["contribution"][expert] += norms[i]
    return [{"domain": d, "layer": l, "phase": p, **row} for (d, l, p), row in sorted(groups.items())]


def select_experts(profiles, keep, metric="frequency", protected=None):
    if metric not in ("frequency", "contribution"):
        raise ValueError("선정 지표는 frequency/contribution이어야 합니다")
    layers = sorted({r["layer"] for r in profiles})
    if not layers:
        raise ValueError("수집된 Routing 통계가 없습니다")
    selected = {}
    for layer in layers:
        scores = None
        for domain in ("math", "code"):
            rows = [r for r in profiles if r["layer"] == layer and r["domain"] == domain]
            if not rows or any(metric == "contribution" and not r["contribution_observed"] for r in rows):
                raise ValueError(f"관측되지 않은 {domain}/{layer}/{metric}")
            total = [sum(r[metric][i] for r in rows) for i in range(len(rows[0][metric]))]
            denom = sum(total)
            if denom <= 0:
                raise ValueError("중요도 합이 0입니다")
            if scores is None:
                scores = [0.] * len(total)
            for i, value in enumerate(total):
                scores[i] += value / denom / 2
        if not 1 <= keep <= len(scores):
            raise ValueError("잘못된 유지 개수")
        mandatory = set((protected or {}).get(layer, []))
        if len(mandatory) > keep or any(i < 0 or i >= len(scores) for i in mandatory):
            raise ValueError("보호 Expert가 예산을 초과하거나 유효하지 않습니다")
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        selected[layer] = sorted(mandatory | set([i for i in order if i not in mandatory][:keep - len(mandatory)]))
    return selected


def compare_logits(a, b, atol=1e-4, rtol=1e-4):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all()
    return {"equivalent": bool(valid and np.allclose(a, b, atol=atol, rtol=rtol)),
            "max_abs_error": float(np.max(np.abs(a - b))) if valid else None,
            "atol": atol, "rtol": rtol}


def routing_similarity(profile_a, profile_b, metric="frequency"):
    """계층별 정규화 분포의 total variation distance; 0은 동일 분포."""
    import numpy as np
    def totals(rows):
        result = {}
        for layer in sorted({r["layer"] for r in rows}):
            vectors = [r[metric] for r in rows if r["layer"] == layer]
            v = np.sum(vectors, axis=0)
            if v.sum() > 0:
                result[layer] = v / v.sum()
        return result
    a, b = totals(profile_a), totals(profile_b)
    return [{"layer": layer, "total_variation": float(np.abs(a[layer] - b[layer]).sum() / 2)} for layer in sorted(a.keys() & b.keys())]


def stability(masks):
    from itertools import combinations
    values = []
    for a, b in combinations(masks, 2):
        for layer in sorted(a.keys() & b.keys()):
            x, y = set(a[layer]), set(b[layer])
            values.append({"layer": layer, "jaccard": len(x & y) / len(x | y)})
    return values


def source_revision():
    def git(*args):
        r = subprocess.run(["git", *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def baseline_environment():
    """2026-11-09: NVML 미관측을 GPU 0개나 유효한 실측으로 오인하지 않는다."""
    import psutil
    from .memory_safety import _ram_sample
    from tools.native.gpu_memory import NvmlDevice
    gpu = None
    try:
        device = NvmlDevice()
        try:
            driver = device.nvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
            gpu = [{'name': device.name, 'uuid': device.uuid,
                    'driver_version': driver, 'total_memory_bytes': device.sample()[0]}]
        finally:
            device.close()
    except Exception:
        pass
    try:
        ram = _ram_sample()[0]
    except Exception:
        ram = None
    cpu_name = platform.processor() or None
    return {'platform': platform.platform(), 'processor': platform.processor(), 'ram_bytes': ram, 'gpu': gpu,
            'cpu': {'name': cpu_name, 'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(logical=True)}}


class Experiment:
    """실제 모델 작업은 명시적인 사용자 셀 호출에서만 실행한다."""
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self._identities = {}
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        digest = canonical_hash(config)
        if path.exists():
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
            if self.manifest["config_hash"] != digest:
                raise ValueError("설정이 변경되었습니다. 새 run_id를 사용하세요")
        else:
            self.manifest = {"schema_version": 1, "config": config, "config_hash": digest,
                             "source": source_revision(), "stages": {}, "created_unix": time.time(),
                             "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                             "processor": platform.processor(), "dependencies": {}}}
            for name in ('numpy', 'psutil', 'datasets', 'evalplus', 'nbformat'):
                try:
                    self.manifest['environment']['dependencies'][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    self.manifest['environment']['dependencies'][name] = None
            write_json(path, self.manifest)

    def record(self, stage, evidence):
        self.manifest["stages"][stage] = evidence
        write_json(self.root / "manifest.json", self.manifest)

    def require(self, *gates):
        for gate in gates:
            if not self.manifest["stages"].get(gate, {}).get("equivalent", False):
                raise RuntimeError(f"실측 동등성 게이트가 필요합니다: {gate}")

    def run(self, name, request, timeout_s=None):
        """2026-11-05: 시간 대신 물리·commit·GPU 메모리 여유를 감시한다."""
        folder = self.root / "jobs" / name
        request = {**request, "seed": request.get("seed", self.config["seed"]),
                   "context_size": request.get("context_size", self.config["context_size"])}
        request = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
        executable = Path(self.config["engine"])
        if not executable.is_file():
            raise FileNotFoundError(f"native 엔진을 먼저 빌드하세요: {executable}")
        identity = {}
        for role, path in (("engine", executable), ("model", Path(request["model"]))):
            stat = path.stat()
            key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            if key not in self._identities:
                self._identities[key] = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}
            identity[role] = self._identities[key]
        if folder.exists():
            saved = json.loads((folder / "request.json").read_text(encoding="utf-8"))
            saved_identity = json.loads((folder / "identity.json").read_text(encoding="utf-8"))
            if saved != request or saved_identity != identity:
                raise ValueError("기존 job의 설정 또는 모델/엔진 해시가 다릅니다")
            if (folder / "measurement.json").exists():
                return self.result(folder)
            raise RuntimeError(f"미완료 job입니다. 원인 확인 후 다른 이름을 사용하세요: {folder}")
        guard = MemoryGuard(self.config, request)
        try:
            try:
                preflight = guard.preflight()
            except Exception as error:
                # 2026-11-05: 재시도를 막는 job 폴더는 만들지 않고 사전 차단 근거만 보존한다.
                try:
                    write_json(self.root / 'preflight-failures' / f'{name}.json',
                               {'reason': str(error), 'error_type': type(error).__name__, 'memory_last': guard.last})
                except Exception as record_error:
                    error.add_note(f'사전 차단 기록 실패: {record_error}')
                raise
            # 2026-11-07: 01의 단일 요청도 02 채점과 같은 실행 경계를 공유한다.
            from .execution_safety import execution_guard
            with execution_guard('native'):
                return self._execute_job(folder, request, identity, executable, guard, preflight, timeout_s)
        finally:
            active_error = sys.exception()
            try:
                guard.close()
            except Exception as close_error:
                if active_error is None:
                    raise
                active_error.add_note(f'메모리 감시 종료 실패: {close_error}')

    def _execute_job(self, folder, request, identity, executable, guard, preflight, timeout_s):
        import psutil
        from tools.native.gpu_memory import ProcessGpuMemorySampler
        folder.mkdir(parents=True)
        write_json(folder / "request.json", request)
        write_json(folder / "identity.json", identity)
        self.manifest.setdefault("artifacts", {}).update({x["path"]: x for x in identity.values()})
        write_json(self.root / "manifest.json", self.manifest)
        started, peak, failure = time.perf_counter(), None, None
        with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
            process = spawn_native([str(executable.resolve()), "--request", str((folder / "request.json").resolve()), "--output", str((folder / "events.jsonl").resolve())], stdout=out, stderr=err)
            monitor, gpu_sampler = None, None
            try:
                gpu_sampler = ProcessGpuMemorySampler(process.pid)
                try:
                    monitor = psutil.Process(process.pid)
                except psutil.NoSuchProcess:
                    # 2026-11-05: 빠른 종료는 실패나 RSS=0이 아니라 미관측으로 처리한다.
                    pass
                while process.poll() is None:
                    guard.check()
                    if monitor is None:
                        # PID가 이미 사라진 경우에는 wait로 종료 상태를 수거한다.
                        break
                    gpu_sampler.sample()
                    try:
                        members = [monitor] + monitor.children(recursive=True)
                    except psutil.NoSuchProcess:
                        break
                    total, sampled = 0, False
                    for member in members:
                        try:
                            total += member.memory_info().rss
                            sampled = True
                        except psutil.NoSuchProcess:
                            continue
                    if sampled:
                        peak = total if peak is None else max(peak, total)
                    if timeout_s is not None and time.perf_counter() - started > timeout_s:
                        raise TimeoutError("실험 job timeout")
                    time.sleep(.1)
                # poll 이전/관측 도중 종료해도 실제 returncode를 반드시 수거한다.
                process.wait()
            except BaseException as exc:
                failure = str(exc)
                cleanup_errors = []
                try:
                    children = monitor.children(recursive=True) if monitor is not None else []
                except psutil.NoSuchProcess:
                    children = []
                except Exception as error:
                    cleanup_errors.append(f'children: {type(error).__name__}: {error}')
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as error:
                        cleanup_errors.append(f'child kill: {type(error).__name__}: {error}')
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    if process.poll() is None:
                        raise RuntimeError('native process remained alive after wait')
                except Exception as error:
                    cleanup_errors.append(f'parent kill/wait: {type(error).__name__}: {error}')
                    from .execution_safety import poison_execution
                    poison_execution('단일 native 작업 종료 확인 실패')
                gpu_measurement = None
                try:
                    if gpu_sampler:
                        gpu_measurement = gpu_sampler.snapshot()
                        gpu_sampler.close()
                except Exception as error:
                    cleanup_errors.append(f'gpu sampler: {type(error).__name__}: {error}')
                try:
                    write_json(folder / "failure.json", {"reason": failure, "peak_ram_bytes": peak, "os_gpu_memory": gpu_measurement,
                                                         "memory_preflight": preflight, "memory_last": guard.last,
                                                         "cleanup_errors": cleanup_errors})
                except Exception as error:
                    exc.add_note(f'실패 기록 저장 실패: {error}')
                raise
            gpu_measurement = gpu_sampler.snapshot()
            gpu_sampler.close()
        if process.returncode:
            write_json(folder / "failure.json", {"returncode": process.returncode, "peak_ram_bytes": peak})
            raise RuntimeError(f"native job 실패: {folder / 'stderr.log'}")
        write_json(folder / "measurement.json", {"peak_process_tree_rss_bytes": peak, "wall_s": time.perf_counter() - started,
                                                "os_gpu_memory": gpu_measurement, "memory_preflight": preflight, "memory_last": guard.last})
        return self.result(folder)

    @staticmethod
    def result(folder):
        rows = [r for r in read_jsonl(Path(folder) / "events.jsonl") if r.get("type") == "result"]
        if len(rows) != 1:
            raise ValueError("native result는 정확히 한 개여야 합니다")
        result = rows[0]
        result["measurement"] = json.loads((Path(folder) / "measurement.json").read_text(encoding="utf-8"))
        return result

    def compare(self, name, model_a, model_b, records, mask_a=None, mask_b=None, instrumentation_a=False, instrumentation_b=False):
        comparisons = []
        for i, item in enumerate(records):
            base = {"prompt": item["prompt"], "mode": "logits", "max_tokens": 0, "instrumentation": False,
                    "n_gpu_layers": self.config["n_gpu_layers"], "apply_chat_template": True}
            a = self.run(f"{name}-a-{i}", {**base, "model": str(model_a), "allowlist": mask_a, "instrumentation": instrumentation_a})
            b = self.run(f"{name}-b-{i}", {**base, "model": str(model_b), "allowlist": mask_b, "instrumentation": instrumentation_b})
            comparisons.append({"id": item["id"], **compare_logits(a.get("logits", []), b.get("logits", []))})
        evidence = {"equivalent": bool(comparisons) and all(r["equivalent"] for r in comparisons), "comparisons": comparisons}
        self.record(name, evidence)
        return evidence

    def collect(self, name, model, records, mode="profile", mask=None, max_tokens=256, instrumentation=None, warmup=False):
        results = []
        for i, item in enumerate(records):
            result = self.run(f"{name}-{i}", {"model": str(model), "prompt": item["prompt"], "mode": mode,
                              "reference": item.get("reference", ""), "max_tokens": max_tokens,
                              "instrumentation": mode == "profile" if instrumentation is None else instrumentation,
                              "warmup": warmup,
                              "allowlist": mask, "apply_chat_template": True,
                              "n_gpu_layers": self.config["n_gpu_layers"]})
            results.append({"id": item["id"], "domain": item["domain"], "dataset": item.get("dataset", "custom"),
                            "prompt": item["prompt"], "reference": item.get("reference"), **result})
        write_json(self.root / f"{name}.json", results)
        return results

    def routing(self, name, records):
        for i, item in enumerate(records):
            for row in read_jsonl(self.root / "jobs" / f"{name}-{i}" / "events.jsonl"):
                if row.get("type") == "routing":
                    yield {**row, "sample_id": item["id"], "domain": item["domain"]}


def analyse_generation(row):
    text = row.get("text", "")
    result = {"id": row["id"], "domain": row["domain"], "dataset": row.get("dataset", "custom"), "empty": not text.strip(),
              "truncated": row.get("stop_reason") == "length", "syntax_valid": None, "math_correct": None,
              "repeated_4gram_ratio": None, "code_execution_pass_rate": None}
    tokens = text.split()
    grams = [tuple(tokens[i:i + 4]) for i in range(max(0, len(tokens) - 3))]
    if grams:
        result["repeated_4gram_ratio"] = 1 - len(set(grams)) / len(grams)
    if row["domain"] == "code":
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        code = blocks[-1] if blocks else text
        try:
            ast.parse(code)
            result["syntax_valid"] = True
        except (SyntaxError, ValueError):
            result["syntax_valid"] = False
    elif row["domain"] == "math" and row.get("reference"):
        # GSM8K 정수/소수 정답만 자동 채점. 기호식 MATH 답은 별도 검토 대상으로 둔다.
        reference = str(row["reference"]).split("####")[-1].replace(",", "").strip()
        answers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        try:
            expected = float(reference)
            result["math_correct"] = bool(answers) and math.isclose(float(answers[-1]), expected, rel_tol=0, abs_tol=1e-8)
            result["math_scoring"] = "last_numeric_answer_exploratory"
        except ValueError:
            result["math_scoring"] = "manual_symbolic_review_required"
    return result


def summarize(results):
    rows = [analyse_generation(r) for r in results]
    summary = {}
    def metrics(group):
        scores = [r["math_correct"] for r in group if r["math_correct"] is not None]
        n, successes = len(scores), sum(scores)
        interval = None
        if n:
            z, p = 1.96, successes / n
            center = (p + z*z/(2*n))/(1+z*z/n)
            half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            interval = [center-half, center+half]
        return {"n": len(group), "numeric_math_scored": n, "numeric_math_excluded": len(group)-n if group[0]['domain'] == 'math' else None,
                           "numeric_math_accuracy": successes/n if n else None,
                           "wilson_95": interval, "code_execution_pass_rate": None,
                           "syntax_valid": sum(r["syntax_valid"] is True for r in group),
                           "empty": sum(r["empty"] for r in group), "truncated": sum(r["truncated"] for r in group)}
    for domain in sorted({r["domain"] for r in rows}):
        summary[domain] = metrics([r for r in rows if r["domain"] == domain])
    by_dataset = {dataset: metrics([r for r in rows if r['dataset'] == dataset]) for dataset in sorted({r['dataset'] for r in rows})}
    return {"summary": summary, "by_dataset": by_dataset, "items": rows,
            "math_metric_scope": "Exploratory last-number heuristic only; mixed numeric subset is not MATH-500 benchmark accuracy"}


def benchmark(exp, name, model, prompts, mask=None, repetitions=5, max_tokens=256):
    all_rows = []
    for repetition in range(repetitions):
        all_rows.extend(exp.collect(f"{name}-r{repetition}", model, prompts, mode="generate", mask=mask, max_tokens=max_tokens, warmup=True))
    metrics = {}
    for key in ("load_s", "ttft_s", "prefill_tokens_s", "decode_tokens_s"):
        values = [r.get("timings", {}).get(key) for r in all_rows]
        values = [v for v in values if v is not None and math.isfinite(v)]
        metrics[key] = {"n": len(values), "median": statistics.median(values) if values else None,
                        "min": min(values) if values else None, "max": max(values) if values else None}
    for key, getter in (('peak_process_tree_rss_bytes', lambda r: r['measurement'].get('peak_process_tree_rss_bytes')),
                        ('nvml_process_gpu_sampled_peak_bytes', lambda r: r['measurement'].get('os_gpu_memory', {}).get('peak_bytes'))):
        values = [getter(r) for r in all_rows]
        values = [v for v in values if v is not None]
        metrics[key] = {'n': len(values), 'median': statistics.median(values) if values else None,
                        'min': min(values) if values else None, 'max': max(values) if values else None}
    write_json(exp.root / f"{name}-benchmark.json", metrics)
    return metrics


def ablation_sensitivity(exp, model, records, mask, full_experts, budget=8):
    """제외 후보별 단일 제거 NLL; 소규모 budget은 전체 Critical Expert 검증이 아니다."""
    reference = exp.collect("sensitivity-baseline", model, records, mode="reference_loss", max_tokens=0)
    baseline = [r.get("reference_nll") for r in reference]
    if not baseline or any(v is None or not math.isfinite(v) for v in baseline):
        raise ValueError("실제 reference NLL이 필요합니다")
    layers = sorted(mask)
    removed = {l: [i for i in range(full_experts) if i not in mask[l]] for l in layers}
    candidates = [(l, removed[l][i]) for i in range(max((len(v) for v in removed.values()), default=0)) for l in layers if i < len(removed[l])][:budget]
    rows = []
    for layer, expert in candidates:
        allow = {l: [i for i in range(full_experts) if not (l == layer and i == expert)] for l in layers}
        # Runtime mask may differ in cardinality by layer; GGUF export cardinality remains uniform.
        result = exp.collect(f"sensitivity-l{layer}-e{expert}", model, records, mode="reference_loss", mask=allow, max_tokens=0)
        losses = [r.get("reference_nll") for r in result]
        if any(v is None or not math.isfinite(v) for v in losses):
            raise ValueError("Ablation NLL 미관측")
        deltas = [b - a for a, b in zip(baseline, losses)]
        rows.append({"layer": layer, "expert": expert, "mean_delta_nll": statistics.mean(deltas), "sample_deltas": deltas})
    write_json(exp.root / "sensitivity.json", rows)
    return rows


def render_tradeoff(points):
    """외부 plotting 의존성 없이 파일 크기–수학 점수 탐색 그림을 만든다."""
    from html import escape
    available = [p for p in points if p.get("accuracy") is not None and p.get("file_bytes")]
    if not available:
        return "<p>아직 비교 가능한 실측 결과가 없습니다.</p>"
    max_size = max(p["file_bytes"] for p in available)
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">',
                '<rect width="720" height="360" fill="white"/><path d="M60 20V300H690" fill="none" stroke="black"/>',
                '<text x="240" y="345">파일 크기 (원본 대비 비율)</text><text x="65" y="18">탐색 수학 정확도 (0–1)</text>']
    for p in available:
        x, y = 60 + 620 * p["file_bytes"] / max_size, 300 - 270 * p["accuracy"]
        elements.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#167D9A"/><text x="{x-40}" y="{y-10}" font-size="12">{escape(p["name"])}</text>')
    return "".join(elements) + "</svg>"


def paired_quality(original, variant, seed=42, bootstrap_samples=2000):
    """동일 문제의 숫자 정답 변화에 대한 paired bootstrap; 코딩 정답 추정 아님."""
    a, b = {r['id']: analyse_generation(r) for r in original}, {r['id']: analyse_generation(r) for r in variant}
    if a.keys() != b.keys():
        raise ValueError('대응 평가 문제 ID가 다릅니다')
    rng = random.Random(seed)
    output = {}
    for dataset in sorted({r['dataset'] for r in a.values()}):
        ids = [key for key, row in a.items() if row['dataset'] == dataset]
        deltas = []
        for key in ids:
            if a[key]['dataset'] != b[key]['dataset']:
                raise ValueError('대응 평가 데이터셋이 다릅니다')
            if a[key]['math_correct'] is not None and b[key]['math_correct'] is not None:
                deltas.append(int(b[key]['math_correct']) - int(a[key]['math_correct']))
        intervals = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(bootstrap_samples)) if deltas else []
        output[dataset] = {'paired_numeric_n': len(deltas), 'delta_accuracy': statistics.mean(deltas) if deltas else None,
                           'paired_bootstrap_95': [intervals[int(.025*(len(intervals)-1))], intervals[int(.975*(len(intervals)-1))]] if intervals else None,
                           'code_execution_delta': None, 'interpretation': 'exploratory numeric-answer heuristic only'}
    return output
