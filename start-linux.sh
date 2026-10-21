#!/usr/bin/env bash
set -eu
task_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
for task_python in python3.11 python3.12; do
  if command -v "$task_python" >/dev/null 2>&1; then
    exec "$task_python" "$task_root/start.py"
  fi
done
printf '%s\n' 'Please install Python 3.11 or 3.12 with the venv module.' >&2
exit 2
