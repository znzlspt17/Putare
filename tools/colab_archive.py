"""표준 라이브러리만 사용하는 Colab 아카이브 검증과 복원."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_archives(root, archives):
    """모든 ZIP과 기존 경로를 먼저 검증하고 누락된 파일만 복원한다."""
    # 2026-11-09: 불완전한 전송과 경로 탈출·기존 데이터 덮어쓰기를 함께 방지한다.
    root = Path(root)
    if root.is_symlink():
        raise ValueError('복원 위치가 심볼릭 링크입니다')
    validated = []
    used = set()
    try:
        for archive_path, expected_hash in archives:
            archive_path = Path(archive_path)
            if not archive_path.is_file():
                raise FileNotFoundError(f'Drive에 ZIP을 업로드하세요: {archive_path}')
            if sha256_file(archive_path) != expected_hash:
                raise ValueError(f'ZIP SHA256 불일치: {archive_path}')
            archive = zipfile.ZipFile(archive_path)
            validated.append((archive, {}))
            names = archive.namelist()
            if len(names) != len(set(names)) or '_putare_inventory.json' not in names:
                raise ValueError('ZIP 파일 목록에 중복 또는 인벤토리 누락')
            inventory = json.loads(archive.read('_putare_inventory.json'))
            if not isinstance(inventory, dict) or set(names) != set(inventory) | {'_putare_inventory.json'}:
                raise ValueError('ZIP 인벤토리 불일치')
            for name, digest in inventory.items():
                path = PurePosixPath(name)
                if (not isinstance(name, str) or path.is_absolute() or not path.parts
                        or any(p in ('', '.', '..') for p in name.split('/')) or '\\' in name or ':' in name):
                    raise ValueError('안전하지 않은 ZIP 경로')
                info = archive.getinfo(name)
                if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('ZIP에 링크 또는 디렉터리가 있습니다')
                if name in used:
                    raise ValueError('소스와 데이터 ZIP에 중복 경로가 있습니다: ' + name)
                used.add(name)
                with archive.open(info) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                        raise ValueError('ZIP 파일 SHA256 불일치: ' + name)
                target = root.joinpath(*path.parts)
                for part in [root, *[root.joinpath(*path.parts[:i]) for i in range(1, len(path.parts) + 1)]]:
                    if part.is_symlink():
                        raise ValueError('복원 경로에 심볼릭 링크가 있습니다: ' + str(part))
                    if part != target and part.exists() and not part.is_dir():
                        raise FileExistsError('복원 폴더 위치가 파일입니다: ' + str(part))
                if target.exists() and (not target.is_file() or sha256_file(target) != digest):
                    raise FileExistsError('기존 파일이 달라 덮어쓰지 않습니다: ' + str(target))
                validated[-1][1][name] = target
        for archive, paths in validated:
            for name, target in paths.items():
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as stream, target.open('xb') as output:
                        shutil.copyfileobj(stream, output)
    except (zipfile.BadZipFile, KeyError, UnicodeError) as error:
        raise ValueError('ZIP이 손상되었습니다') from error
    finally:
        for archive, _ in validated:
            archive.close()
    return len(used)
