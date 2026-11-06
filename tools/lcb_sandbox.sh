#!/bin/sh
# 2026-11-06: venv 밖의 system Python으로 namespace 경계를 먼저 만든다.
set -eu
task_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /usr/bin/python3 "$task_script_dir/lcb_sandbox.py" "$@"
