"""Validate quoted Python against the evidence supplied for this answer."""
import re
import textwrap


def _same_indented_window(block, packets):
    """Allow a uniform display dedent, retaining relative Python indentation."""
    expected = textwrap.dedent(block).strip('\n')
    size = len(expected.splitlines())
    if not size:
        return False
    for packet in packets:
        lines = packet.splitlines()
        for start in range(len(lines) - size + 1):
            window = '\n'.join(lines[start:start + size])
            if textwrap.dedent(window).strip('\n') == expected:
                return True
    return False


def checked_source_quotes(answer, context):
    packets = []
    def collect(value):
        if isinstance(value, dict):
            if value.get('file'):
                for key in ('code','excerpt','excerpts'):
                    values=value.get(key)
                    values=[values] if isinstance(values,str) else values if isinstance(values,list) else []
                    packets.extend((str(value['file']),packet) for packet in values if isinstance(packet,str))
            for child in value.values(): collect(child)
        elif isinstance(value,list):
            for child in value: collect(child)
    collect(context)
    by_file={};conflicts=set();plain=[]
    for filename,packet in packets:
        lines=by_file.setdefault(filename,{})
        for match in re.finditer(r'(?m)^L(\d+): ?(.*)$',packet):
            number=int(match[1]);line=match[2].rstrip()
            if number in lines and lines[number]!=line:conflicts.add((filename,number))
            lines[number]=line
        if not re.search(r'(?m)^L\d+:',packet):plain.append(packet)
    for filename,number in conflicts:by_file[filename].pop(number,None)
    # Loss and optimizer statements may be supplied as adjacent packets. Merge
    # only consecutive line numbers belonging to the same exact file identity.
    for lines in by_file.values():
        chunk=[];previous=None
        for number,line in sorted(lines.items()):
            if previous is not None and number!=previous+1:
                plain.append('\n'.join(chunk));chunk=[]
            chunk.append(line);previous=number
        if chunk:plain.append('\n'.join(chunk))
    rejected = 0
    def verify(match):
        nonlocal rejected
        block = match[1].strip('\n')
        cited = list(re.finditer(r'(?m)^L(\d+): ?(.*)$',block))
        if cited:
            valid = any(all(lines.get(int(row[1])) == row[2].rstrip() for row in cited) for lines in by_file.values())
            valid = valid and len(cited) == len([line for line in block.splitlines() if line.strip()])
            valid = valid and [int(row[1]) for row in cited]==sorted({int(row[1]) for row in cited})
        else:
            valid = _same_indented_window(block, plain)
        if valid: return match[0]
        rejected += 1
        return '> 此处源码引文与本次读取的原文不一致，已移除；不能据此判断代码实现。'
    result = re.sub(r'```(?:python|py)\s*\n(.*?)```',verify,answer,flags=re.S|re.I)
    return result, {'checked_python_blocks':len(re.findall(r'```(?:python|py)\s*\n',answer,re.I)),
                    'rejected_blocks':rejected, 'scope':'仅核对源码引文，不验证全部科学解释'}
