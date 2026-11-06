"""2026-11-06: 공식 채점기를 사설 Linux root/network/PID namespace에서 실행한다.

venv는 패키지 분리만 제공한다. 이 launcher는 생성 코드에 호스트의 /home,
/mnt 및 인증 저장소를 노출하지 않으며 namespace가 불가능하면 실행하지 않는다.
"""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile


def worker_environment():
    return {'PATH': '/venv/bin:/usr/bin:/bin', 'HOME': '/tmp', 'TMPDIR': '/tmp',
            'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPYCACHEPREFIX': '/tmp/pycache',
            'OPENBLAS_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}


def namespace_command(work, source, venv, adapter):
    unshare = shutil.which('unshare')
    if not unshare:
        raise RuntimeError('unshare가 없어 생성 코드를 실행하지 않습니다')
    return [unshare, '--user', '--map-root-user', '--mount', '--pid', '--fork',
            '--net', '--ipc', '--uts', '--kill-child', sys.executable,
            str(Path(__file__).resolve()), '--child', str(work), str(source),
            str(venv), str(adapter)]


def read_result(path, maximum_bytes=8 * 1024**2):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('채점 결과가 symlink입니다')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            raise ValueError('채점 결과 크기 또는 파일 형식이 잘못되었습니다')
        result = stream.read(maximum_bytes + 1)
    if len(result) > maximum_bytes:
        raise ValueError('채점 결과 크기 초과')
    return result


def _mount(source, target, filesystem=None, flags=0, data=None):
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.mount
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                   ctypes.c_ulong, ctypes.c_char_p]
    fn.restype = ctypes.c_int
    encode = lambda value: os.fsencode(value) if value is not None else None
    if fn(encode(source), encode(target), encode(filesystem), flags, encode(data)):
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(target))


def _readonly_bind(source, target, device=False):
    source, target = Path(source), Path(target)
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    # /usr 하위 WSL mount도 같은 namespace에 포함해야 locked mount를 누락하지 않는다.
    _mount(source, target, flags=4096 | (16384 if source.is_dir() else 0))

    class MountAttr(ctypes.Structure):
        _fields_ = [('attr_set', ctypes.c_uint64), ('attr_clr', ctypes.c_uint64),
                    ('propagation', ctypes.c_uint64), ('userns_fd', ctypes.c_uint64)]

    attributes = MountAttr(1 | 2 | (0 if device else 4), 0, 0, 0)
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'mount_setattr', None)
    if function is None:
        raise RuntimeError('recursive readonly mount를 지원하는 glibc가 필요합니다')
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
                         ctypes.POINTER(MountAttr), ctypes.c_size_t]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(target), 0x8000 if source.is_dir() else 0,
                ctypes.byref(attributes), ctypes.sizeof(attributes)):
        number = ctypes.get_errno()
        raise OSError(number, 'recursive readonly mount 실패', str(target))


def _drop_capabilities():
    # 2026-11-06: namespace root도 mount/chroot를 복구하지 못하도록 권한을 버린다.
    libc = ctypes.CDLL(None, use_errno=True)
    for capability in range(64):
        if libc.prctl(24, capability, 0, 0, 0) != 0 and ctypes.get_errno() != errno.EINVAL:
            raise OSError(ctypes.get_errno(), 'capability bounding set drop 실패')

    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32),
                    ('inheritable', ctypes.c_uint32)]

    header, values = Header(0x20080522, 0), (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(values)) != 0:
        raise OSError(ctypes.get_errno(), 'capset 실패')
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'no_new_privs 실패')


def child_main(work, source, venv, adapter):
    import resource

    work, source, venv, adapter = (Path(p).resolve(strict=True) for p in (work, source, venv, adapter))
    root = work / 'root'
    root.mkdir()
    _mount(None, '/', flags=16384 | (1 << 18))  # MS_REC|MS_PRIVATE
    for name in ('usr', 'bin', 'lib', 'lib64'):
        origin, target = Path('/') / name, root / name
        if origin.is_symlink():
            target.symlink_to(os.readlink(origin))
        elif origin.exists():
            _readonly_bind(origin, target)
    for origin, target in ((source, root / 'source'), (venv, root / 'venv'),
                           (adapter, root / 'app/tools/lcb_grade.py'),
                           (work / 'input.json', root / 'input.json')):
        _readonly_bind(origin, target)
    if Path('/etc/ld.so.cache').is_file():
        _readonly_bind('/etc/ld.so.cache', root / 'etc/ld.so.cache')
    for name in ('null', 'zero', 'random', 'urandom'):
        _readonly_bind(Path('/dev') / name, root / 'dev' / name, device=True)
    for name, size in (('tmp', '512m'), ('out', '32m'), ('dev/shm', '64m')):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        _mount('tmpfs', directory, 'tmpfs', 2 | 4, f'size={size},mode=1777')
    (root / 'proc').mkdir()
    _mount('proc', root / 'proc', 'proc', 2 | 4 | 8)  # NOSUID|NODEV|NOEXEC
    # 루트의 mount point 외에는 새 host-backed 파일을 생성하지 못한다.
    root.chmod(0o555)
    os.chroot(root)
    os.chdir('/tmp')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
    _drop_capabilities()
    with open('/tmp/worker.stdout', 'wb') as out, open('/tmp/worker.stderr', 'wb') as err:
        result = subprocess.run(['/venv/bin/python', '/app/tools/lcb_grade.py', '--worker',
                                 '--official-source', '/source', '--input', '/input.json',
                                 '--output', '/out/result.json'], env=worker_environment(),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True)
    if result.returncode:
        # 사용자 코드/로그를 그대로 내보내지 않고 원인 분류용 종료 코드만 전달한다.
        raise RuntimeError(f'격리 채점 worker 실패: returncode={result.returncode}')
    payload = read_result('/out/result.json')
    json.loads(payload)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def run(source, venv, input_path, output_path, adapter=None, timeout_s=7200):
    if sys.platform != 'linux':
        raise RuntimeError('이 launcher는 WSL Linux에서 실행해야 합니다')
    source, venv, input_path = (Path(p).resolve(strict=True) for p in (source, venv, input_path))
    adapter = Path(adapter or Path(__file__).with_name('lcb_grade.py')).resolve(strict=True)
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('기존 채점 결과를 덮어쓰지 않습니다')
    with tempfile.TemporaryDirectory(prefix='wemakemodel-lcb-') as directory:
        work = Path(directory)
        shutil.copyfile(input_path, work / 'input.json')
        try:
            with (work / 'stdout.json').open('wb') as out, (work / 'stderr.log').open('wb') as err:
                subprocess.run(namespace_command(work, source, venv, adapter), check=True,
                               env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                               stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                               close_fds=True, timeout=timeout_s)
        except subprocess.CalledProcessError as error:
            # worker 로그는 내부 tmpfs에 남고 여기에는 trusted launcher 오류만 있다.
            details = (work / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-2048:]
            raise RuntimeError(f'격리 채점 launcher 실패: {details}') from error
        payload = read_result(work / 'stdout.json')
        json.loads(payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('source')
    parser.add_argument('venv')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    if args.child:
        # 내부 호출에서는 positional 값이 work/source/venv/adapter이다.
        child_main(args.source, args.venv, args.input, args.output)
    else:
        run(args.source, args.venv, args.input, args.output)


if __name__ == '__main__':
    main()
