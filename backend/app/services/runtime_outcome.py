"""A swallowed Python exception can coexist with an OS exit code of zero."""
import re

MARKER = '[ReproMindReportedFailure] process_exit_code=0'


def reported_exception(output):
    start = output.rfind('Traceback (most recent call last):')
    if start < 0:
        return None
    matches = list(re.finditer(r'(?m)^([A-Za-z_][\w.]*(?:Error|Exception)[\w.]*)[ \t]*:[ \t]*(.*)$', output[start:]))
    if not matches:
        return None
    match = matches[-1]
    return {'kind': match[1], 'detail': match[2], 'end': start + match.end()}


def process_exit_code(effective_code, output):
    return 0 if MARKER in output else effective_code
