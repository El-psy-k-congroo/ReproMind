"""Validate quoted Python against the evidence supplied for this answer."""
import re


def checked_source_quotes(answer, context):
    packets = []
    def collect(value):
        if isinstance(value, dict):
            if value.get('file'):
                for key in ('code','excerpt'):
                    if isinstance(value.get(key),str):
                        packets.append(value[key])
            for child in value.values(): collect(child)
        elif isinstance(value,list):
            for child in value: collect(child)
    collect(context)
    numbered = []
    for packet in packets:
        numbered.append({int(m[1]):m[2].rstrip() for m in re.finditer(r'(?m)^L(\d+): ?(.*)$',packet)})
    rejected = 0
    def verify(match):
        nonlocal rejected
        block = match[1].strip('\n')
        cited = list(re.finditer(r'(?m)^L(\d+): ?(.*)$',block))
        if cited:
            valid = any(all(lines.get(int(row[1])) == row[2].rstrip() for row in cited) for lines in numbered)
            valid = valid and len(cited) == len([line for line in block.splitlines() if line.strip()])
        else:
            valid = any(block in re.sub(r'(?m)^L\d+: ?', '',packet) for packet in packets)
        if valid: return match[0]
        rejected += 1
        return '> 此处源码引文与本次读取的原文不一致，已移除；不能据此判断代码实现。'
    result = re.sub(r'```(?:python|py)\s*\n(.*?)```',verify,answer,flags=re.S|re.I)
    return result, {'checked_python_blocks':len(re.findall(r'```(?:python|py)\s*\n',answer,re.I)),
                    'rejected_blocks':rejected, 'scope':'仅核对源码引文，不验证全部科学解释'}
