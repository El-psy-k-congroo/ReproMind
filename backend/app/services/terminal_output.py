"""Plain-text presentation of terminal output; raw operation files stay intact."""
import re


_ESCAPE = re.compile(r'\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-_])')


def readable_output(value: str) -> str:
    value = _ESCAPE.sub('', value or '').replace('\r\n', '\n').replace('\r', '\n')
    if not any(ord(char) < 32 and char not in '\n\t' for char in value):
        return value
    output: list[str] = []
    for char in value:
        if char == '\b':
            if output and output[-1] != '\n':
                output.pop()
        elif char in '\n\t' or ord(char) >= 32:
            output.append(char)
    return ''.join(output)
