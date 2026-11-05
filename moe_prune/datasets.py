"""공개 데이터셋 다운로드는 사용자가 실행한 셀에서만 발생한다."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import urllib.request

from .research import assert_disjoint, split_records, write_json, write_jsonl, prompt_hash, canonical_hash, read_jsonl
from .gguf import sha256


def _revision(repo):
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=30) as response:
        info = json.load(response)
    return info["sha"], info.get("cardData", {}).get("license")


def prepare_datasets(destination, calibration=64, selection=64, evaluation_limit=64, seed=42):
    """2026-11-05: 현재 revision을 해소하고 실제 읽은 revision/표본 ID를 고정한다."""
    from datasets import load_dataset
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("데이터 snapshot이 이미 있습니다. 재사용하거나 새 경로를 지정하세요")
    repos = {"gsm8k": ("openai/gsm8k", "main"), "mbpp": ("google-research-datasets/mbpp", "full"),
             "math500": ("HuggingFaceH4/MATH-500", None), "general": ("Salesforce/wikitext", "wikitext-2-raw-v1")}
    resolved = {name: _revision(repo) for name, (repo, _) in repos.items()}
    def load(name, split):
        repo, subset = repos[name]
        return load_dataset(repo, subset, split=split, revision=resolved[name][0])
    groups = {"calibration": [], "selection": [], "evaluation": []}
    for name, domain, prompt_key, ref_key in (("gsm8k", "math", "question", "answer"), ("mbpp", "code", "text", "code")):
        records = [{"id": f"{name}:train:{r.get('task_id', i)}", "domain": domain, "dataset": name,
                    "prompt": r[prompt_key], "reference": r[ref_key]} for i, r in enumerate(load(name, "train"))]
        split = split_records(records, calibration, selection, seed)
        groups["calibration"].extend(split["calibration"])
        groups["selection"].extend(split["selection"])
    general = []
    seen = set()
    for i, row in enumerate(load("general", "train")):
        text = row["text"].strip()
        digest = prompt_hash(text)
        if len(text.split()) >= 40 and digest not in seen:
            general.append({"id": f"wikitext:train:{i}", "domain": "general", "dataset": "wikitext", "prompt": text, "reference": ""})
            seen.add(digest)
    general_split = split_records(general, calibration, selection, seed)
    groups["calibration"].extend(general_split["calibration"])
    groups["selection"].extend(general_split["selection"])
    groups["evaluation"].extend(general_split["evaluation"][:evaluation_limit])
    for name, prompt_key, ref_key in (("gsm8k", "question", "answer"), ("math500", "problem", "answer")):
        rows = load(name, "test")
        for i, row in enumerate(rows):
            if evaluation_limit is not None and i >= evaluation_limit:
                break
            groups["evaluation"].append({"id": f"{name}:test:{i}", "domain": "math", "dataset": name,
                                          "prompt": row[prompt_key], "reference": row[ref_key]})
    for name, data in (("humanevalplus", get_human_eval_plus()), ("mbppplus", get_mbpp_plus())):
        for i, (key, row) in enumerate(sorted(data.items())):
            if evaluation_limit is not None and i >= evaluation_limit:
                break
            groups["evaluation"].append({"id": f"{name}:{key}", "domain": "code", "dataset": name,
                                          "prompt": row["prompt"], "reference": row.get("canonical_solution", "")})
    # Cross-benchmark duplicate prompts must be explicit, not silently leak into training/selection.
    assert_disjoint(groups)
    manifest = {"seed": seed, "datasets": {name: {"repo": repos[name][0], "revision": revision, "license": license}
                                             for name, (revision, license) in resolved.items()},
                "evalplus_version": importlib.metadata.version("evalplus"),
                "evalplus_license": "Apache-2.0; dataset licenses must be checked separately",
                "evalplus_source": "https://github.com/evalplus/evalplus/tree/v0.3.1/evalplus/data",
                "full_record_hashes": {name: canonical_hash(rows) for name, rows in groups.items()},
                "samples": {name: [{"id": r["id"], "prompt_sha256": prompt_hash(r["prompt"])} for r in rows] for name, rows in groups.items()}}
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in groups.items():
        write_jsonl(destination / f"{name}.jsonl", rows)
    manifest['file_sha256'] = {name: sha256(destination / f'{name}.jsonl') for name in groups}
    write_json(destination / "manifest.json", manifest)
    return groups, manifest


def load_snapshot(destination):
    destination = Path(destination)
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    groups = {}
    for name in ('calibration', 'selection', 'evaluation'):
        path = destination / f'{name}.jsonl'
        if sha256(path) != manifest['file_sha256'][name]:
            raise ValueError(f'데이터 snapshot hash mismatch: {name}')
        groups[name] = list(read_jsonl(path))
        if canonical_hash(groups[name]) != manifest['full_record_hashes'][name]:
            raise ValueError(f'문제/참조 답변 snapshot 변경: {name}')
    assert_disjoint(groups)
    return groups, manifest
