"""Start the workbench from a source checkout or a portable submission package."""
from __future__ import annotations
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print('ReproMind requires Python 3.11 or 3.12.', file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parent
    runtime = root / '.venv'
    python = runtime / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    requirements = root / 'backend/requirements.txt'
    stamp = runtime / '.repromind-requirements.sha256'
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(runtime)], check=True)
    version = subprocess.check_output([str(python), '-c',
        'import sys;print(str(sys.version_info.major)+"."+str(sys.version_info.minor))'], text=True).strip()
    if version not in {'3.11', '3.12'}:
        print('Existing .venv uses unsupported Python. Select another folder; existing files were preserved.', file=sys.stderr)
        return 2
    if not stamp.exists() or stamp.read_text(encoding='ascii').strip() != digest:
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        stamp.write_text(digest + '\n', encoding='ascii')
    config = root / '.env'
    if not config.exists():
        shutil.copy2(root / '.env.example', config)
        print('Created .env. Configure the model API before using AI features.')
    if not (root / 'frontend/dist/index.html').exists():
        print('Build frontend first: cd frontend; npm ci; npm run build', file=sys.stderr)
        return 2
    print('Starting ReproMind. Default local address: http://127.0.0.1:8010')
    return subprocess.call([str(python), 'run_server.py'], cwd=root / 'backend')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f'Setup failed (exit {exc.returncode}). Correct the dependency or network error and retry.', file=sys.stderr)
        raise SystemExit(exc.returncode)
    except KeyboardInterrupt:
        print('\nReproMind stopped.')
        raise SystemExit(130)
