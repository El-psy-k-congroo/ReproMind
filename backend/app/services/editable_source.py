"""Recognize source/configuration files without exposing arbitrary text assets."""
import re
from pathlib import Path
from packaging.requirements import Requirement, InvalidRequirement


def requirements_file(name):
    return bool(re.fullmatch(r'(?:requirements(?:[-_.][\w.-]+)?|constraints(?:[-_.][\w.-]+)?)\.txt', Path(name).name, re.I))


def editable_file(name):
    return Path(name).suffix.lower() in {'.py', '.yaml', '.yml', '.toml', '.json'} or requirements_file(name)


def validate_requirements(content):
    """Syntax only. Installation and platform compatibility remain runtime checks."""
    logical = content.replace('\\\n', '')
    for number, raw in enumerate(logical.splitlines(), 1):
        line = re.split(r'\s+#', raw, maxsplit=1)[0].strip()
        if not line or line.startswith(('#', '-r ', '--requirement ', '-c ', '--constraint ', '--index-url ', '--extra-index-url ', '--find-links ', '--trusted-host ', '--only-binary ', '--no-binary ')):
            continue
        if line.startswith(('-e ', '--editable ', 'git+', 'http://', 'https://', './', '../')):
            continue  # pip supports VCS/local references; never execute them here.
        try:
            Requirement(re.split(r'\s+--hash=', line, maxsplit=1)[0])
        except InvalidRequirement as exc:
            raise ValueError(f'requirements L{number}: {exc}') from exc
