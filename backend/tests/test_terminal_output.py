import asyncio
import sys

from app.services.terminal_output import readable_output
from app.services import reproduction_engine as engine


def test_conda_spinner_is_not_rendered_as_garbled_characters():
    raw = 'Retrieving notices: - \b\b\\ \b\b| \b\b/ \b\bdone\r\n中文日志\n'
    assert readable_output(raw) == 'Retrieving notices: done\n中文日志\n'


def test_removes_terminal_escapes_without_removing_errors_or_unicode():
    raw = '\x1b[31mSSL error\x1b[0m\x00\x1b]0;title\x07\n中文\ttext\b!'
    assert readable_output(raw) == 'SSL error\n中文\ttex!'
    assert readable_output('a\n\bB') == 'a\nB'


def test_subprocess_disables_nonessential_network_and_progress(tmp_path):
    script = "import os; assert os.environ['CONDA_NUMBER_CHANNEL_NOTICES']=='0'; assert os.environ['CONDA_QUIET']=='true'; assert os.environ['PIP_DISABLE_PIP_VERSION_CHECK']=='1'; print('step: - \\b\\bdone 中文')"
    code, output = asyncio.run(engine._run_process([sys.executable, '-c', script], tmp_path, 10))
    assert code == 0 and 'step: done 中文' in output
    assert '\b' not in output
