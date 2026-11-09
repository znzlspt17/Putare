"""2026-11-06: 고정 revision의 v5 ID를 대용량 사본 없이 직접 검증한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

ID_PATTERN = re.compile(rb'(?:^|[,{])\s*"question_id"\s*:\s*"([^"\\]*)"')


def scan_stream(stream, chunk_size=1024 * 1024):
    """2026-11-06: 최대 512바이트 overlap으로 chunk 경계 ID를 중복 없이 수집한다."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    tail, ids, matches, rows, total = b'', set(), 0, 0, 0
    digest = hashlib.sha256()
    last = b''
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
        total += len(chunk)
        rows += chunk.count(b'\n')
        last = chunk[-1:]
        data = tail + chunk
        previous = len(tail)
        for match in ID_PATTERN.finditer(data):
            if match.end() <= previous:
                continue
            value = match.group(1).decode('utf-8')
            matches += 1
            if value in ids:
                raise ValueError('기존 dataset 파일 내 question_id 중복')
            ids.add(value)
        tail = data[-512:]
    if total and last != b'\n':
        rows += 1
    if matches != rows:
        raise ValueError(f'question_id 수와 행 수가 다릅니다: {matches} != {rows}')
    return ids, rows, digest.hexdigest()


def _open(url):
    # 인증 저장소와 프록시 인증값을 참조하지 않는 공개 HTTPS 요청이다.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(urllib.request.Request(url, headers={'User-Agent': 'putare-benchmark/1'}), timeout=120)


def verify(snapshot, output):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('기존 검증 결과는 덮어쓰지 않습니다')
    manifest_path = snapshot / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    lcb = manifest['datasets']['lcb']
    revision = lcb['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('고정 dataset revision이 유효하지 않습니다')
    source = lcb['source_sha256']
    index_revision, space = source['v5_ids_source_revision'], source['v5_ids_source_space']
    if not re.fullmatch(r'[0-9a-f]{40}', index_revision) or space != 'livecodebench/code_generation_samples':
        raise ValueError('공식 v5 index 출처가 유효하지 않습니다')
    index_url = f'https://huggingface.co/spaces/{space}/resolve/{index_revision}/problems.json'
    with _open(index_url) as response:
        index_bytes = response.read()
    if hashlib.sha256(index_bytes).hexdigest() != source['v5_ids_source_sha256']:
        raise ValueError('공식 v5 index 해시가 snapshot과 다릅니다')
    index = json.loads(index_bytes)
    expected = {str(row['question_id']) for row in index}
    if len(index) != 880 or len(expected) != 880:
        raise ValueError('공식 v5 index ID 개수가 다르거나 중복됩니다')
    added = set()
    with (snapshot / 'records.jsonl').open('rb') as stream:
        digest = hashlib.sha256()
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if row['domain'] == 'code':
                value = str(row['lcb_problem']['question_id'])
                if value in added:
                    raise ValueError('snapshot 신규 ID 중복')
                added.add(value)
    if digest.hexdigest() != manifest['file_sha256']['records'] or len(added) != 175:
        raise ValueError('snapshot 해시 또는 신규 ID 개수가 다릅니다')
    observed, files = set(), {}
    start = time.monotonic()
    for name in source['release_v5_files']:
        if name not in ['test.jsonl'] + [f'test{i}.jsonl' for i in range(2, 6)]:
            raise ValueError('예상 외 v5 파일입니다')
        url = f'https://huggingface.co/datasets/{lcb["repo"]}/resolve/{revision}/{name}'
        with _open(url) as stream:
            ids, count, sha = scan_stream(stream)
        if observed & ids:
            raise ValueError('서로 다른 v5 파일에 동일 question_id가 있습니다')
        observed.update(ids)
        files[name] = {'rows': count, 'sha256': sha}
        print(f'{name}: {count}개 ID 확인', flush=True)
    if observed != expected or observed & added:
        raise ValueError('고정 dataset의 v5 ID와 공식 index가 다르거나 v6 신규 ID와 겹칩니다')
    proof = {'status': 'verified', 'snapshot_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
             'snapshot_records_sha256': manifest['file_sha256']['records'],
             'dataset_revision': revision, 'v5_index_revision': index_revision,
             'v5_count': len(observed), 'v6_new_count': len(added), 'v5_files': files,
             'elapsed_seconds': time.monotonic() - start}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'v5 {len(observed)}개와 신규 {len(added)}개 검증 완료: {output}', flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    verify(arguments.snapshot, arguments.output)
