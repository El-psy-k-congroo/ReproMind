"""Bounded, source-grounded excerpts with reserved space for implementation details."""

import re


DETAILS = re.compile(
    r"fine[-\s]?tun|learning\s+rate|epochs?\b|batch\s+size|train.*(?:split|test)|"
    r"preprocess|hyperparameter|implementation|prompt\s+template|random\s+seed|"
    r"微调|学习率|训练轮|预处理|数据划分", re.I,
)
HEADING = re.compile(r"(?m)^[ \t#]*(?:(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?[ \t]+)?([^\n]{3,100})$")


def core_implementation_concerns(limits: list[str], required_names: list[str] | None = None) -> list[str]:
    """Triage hints only. Negation and the missing component must share a clause.

    'ICL is not implemented; only fine-tuning is supported' must not become
    'fine-tuning is missing'. Scope-specific requirements come from the paper
    specification, not a global list of every extension mentioned in a paper.
    """
    negative = re.compile(r"not (?:include|implement|support|provide)|does not|lacks?|missing|without|未实现|缺少|没有实现", re.I)
    component = re.compile(r"fine.?tun|training|core (?:method|algorithm)|split|微调|训练流程|核心方法|划分", re.I)
    names = [re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", re.I) for name in (required_names or [])]
    concerns = []
    for value in limits:
        for clause in re.split(r"[;；。,]|\b(?:but|however|whereas|only)\b|但是|不过", value, flags=re.I):
            if not negative.search(clause) or not (component.search(clause) or any(pattern.search(clause) for pattern in names)):
                continue
            if re.search(r"checkpoint|api key|download|access|检查点|密钥|下载|访问权限", clause, re.I) and not re.search(
                    r"procedure|implementation|流程|实现", clause, re.I):
                continue
            concerns.append(value)
            break
    return concerns


def _sections(text: str) -> list[tuple[int, int, str]]:
    headings = []
    for match in HEADING.finditer(text):
        title = match.group(2).strip()
        compact = re.sub(r"[^a-z]", "", title.lower())
        numbered = bool(match.group(1))
        if not ((numbered and title.isupper()) or compact in {
            "introduction", "method", "methodology", "experiments", "references",
            "experimentalsetup", "implementation", "appendix", "conclusion",
        } or numbered and re.match(r"(?i)^(method|experiment|implementation|appendix)\b", title)):
            continue
        headings.append((match.start(), compact))
    return [(start, headings[i + 1][0] if i + 1 < len(headings) else len(text), title)
            for i, (start, title) in enumerate(headings)]


def _pack(text: str, spans: list[tuple[int, int]], limit: int) -> str:
    selected = []
    for start, end in spans:
        # Keep exact source quotes and avoid spending the budget on overlaps.
        fragments = [(max(0, start), min(len(text), end))]
        for a, b in selected:
            fragments = [(x, y) for s, e in fragments for x, y in
                         ((s, min(e, a)), (max(s, b), e)) if x < y]
        for a, b in fragments:
            remaining = limit - sum(y - x for x, y in selected) - max(0, len(selected)) * 35
            if remaining <= 0:
                break
            selected.append((a, min(b, a + remaining)))
    return "\n\n[... excerpt gap ...]\n\n".join(text[a:b] for a, b in sorted(selected))[:limit]


def paper_context(text: str, limit: int = 24000) -> str:
    if len(text) <= limit:
        return text
    sections = _sections(text)
    excluded = [(a, b) for a, b, title in sections if title in {"references", "bibliography"}]
    spans = [(0, min(800, limit // 8))]
    # Reserve ~one third of the budget BEFORE long method sections. Appendix
    # fine-tuning/data settings cannot be displaced by bibliography or tables.
    detail_budget = limit // 3
    matches = []
    for match in DETAILS.finditer(text):
        if any(a <= match.start() < b for a, b in excluded):
            continue
        vicinity = text[max(0, match.start() - 200):match.end() + 650]
        numeric_settings = bool(re.search(r"\d", vicinity)) and bool(re.search(
            r"epochs?|learning\s+rate|batch\s+size|random\s+seed", vicinity, re.I))
        appendix = any(a <= match.start() < b and any(word in title for word in
                       ("setting", "finetun", "implement", "preprocess")) for a, b, title in sections)
        matches.append((int(numeric_settings) * 10 + int(appendix) * 5, match))
    for _, match in sorted(matches, key=lambda row: (-row[0], row[1].start())):
        start = text.rfind("\n", 0, max(0, match.start() - 250)) + 1
        end = text.find("\n", min(len(text), match.end() + 650))
        end = len(text) if end < 0 else end
        if any(a <= match.start() < b for a, b in spans):
            continue
        if detail_budget <= 0:
            break
        end = min(end, start + detail_budget)
        spans.append((start, end))
        detail_budget -= end - start
    relevant = [(a, b, title) for a, b, title in sections
                if not any(word in title for word in
                           ("introduction", "background", "relatedwork", "conclusion", "references"))]
    if relevant:
        quota = max(500, (limit * 2 // 3 - 1000) // len(relevant))
        for a, b, _ in relevant:
            spans.extend([(a, min(b, a + quota * 3 // 4)), (max(a, b - quota // 4), b)])
    else:
        spans.append((800, limit // 2))
    return _pack(text, spans, limit)


def related_evidence(text: str, queries: list[str], limit: int = 7000) -> str:
    """Retrieve exact source windows for a failed review, never fabricate quotes."""
    words = set(re.findall(r"[a-zA-Z][a-zA-Z_-]{3,}", " ".join(queries).lower()))
    words -= {"paper", "code", "method", "missing", "does", "with", "from", "that", "this", "file"}
    rows = []
    excluded = [(a, b) for a, b, title in _sections(text) if title in {"references", "bibliography"}]
    for match in re.finditer(r"(?m)^.*$", text):
        if any(a <= match.start() < b for a, b in excluded):
            continue
        line = match.group().lower()
        score = sum(word in line for word in words) + (2 if DETAILS.search(line) else 0)
        if re.search(r"\d", line) and re.search(r"epochs?|learning\s+rate|batch\s+size", line, re.I):
            score += 5
        if score:
            rows.append((score, match.start(), match.end()))
    spans = [(max(0, a - 300), min(len(text), b + 600))
             for _, a, b in sorted(rows, key=lambda row: (-row[0], row[1]))[:12]]
    return _pack(text, spans, limit)
