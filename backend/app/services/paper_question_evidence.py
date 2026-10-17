"""Retrieve exact paper passages with topic coverage, separately from code evidence."""
import json
import re
import unicodedata

TOPICS = [
    (('方法', '原理', '机制', 'method'), ('method', 'approach', 'algorithm', 'framework')),
    (('分类', 'classifier', 'head'), ('classification head', 'classifier', 'logistic regression')),
    (('对比', '句对', 'contrastive', 'siamese'), ('contrastive', 'siamese', 'pairs', 'positive', 'negative')),
    (('采样', '样本', 'few-shot', 'fewshot', 'sampling'), ('sampling', 'samples per class', 'labeled', 'training set')),
    (('损失', '公式', 'loss', 'cosine'), ('loss', 'objective', 'cosine', 'similarity')),
    (('参数', '学习率', '轮数', 'epoch', 'batch', 'learning rate'), ('learning rate', 'batch size', 'epoch', 'hyperparameter')),
    (('结论', '对照', '消融', '结果', 'ablation'), ('comparison', 'ablation', 'results', 'baseline')),
    (('数据', '划分', 'dataset', 'split'), ('dataset', 'split', 'train', 'validation', 'test')),
]


def paper_question_passages(text, question, filename='', limit=5):
    lines = text.splitlines()
    normalized = [unicodedata.normalize('NFKC', line).casefold() for line in lines]
    query = unicodedata.normalize('NFKC', question).casefold()
    groups = [words for triggers, words in TOPICS if any(term in query for term in triggers)]
    terms = list(dict.fromkeys(re.findall(r'[a-z][a-z0-9_-]{2,}', query)))[:20]
    if not groups and not terms:
        return []
    # Bibliography mentions do not establish the current method; appendices
    # after a bibliography remain eligible for implementation details.
    excluded, in_references = set(), False
    sections, section = [], ''
    for i, line in enumerate(normalized):
        if re.fullmatch(r'\s*(?:\d+\s+)?(?:references|bibliography)\s*', line):
            in_references = True
        if re.match(r'^\s*(?:[a-z](?:\.\d+)*\.?\s+)?appendi(?:x|ces)\b', line):
            in_references = False
        if in_references:
            excluded.add(i)
        if re.match(r'^\s*(?:\d+(?:\.\d+)*|[a-z](?:\.\d+)+)\s+[a-z]', line):
            section = line
        sections.append(section)
    candidates = []
    for index in range(len(lines)):
        if index in excluded:
            continue
        if not any(word in normalized[index] for words in groups for word in words) and not any(
                term in normalized[index] for term in terms):
            continue
        start, end = max(0, index - 3), min(len(lines), index + 7)
        while start < index and start in excluded:
            start += 1
        end = next((i for i in range(index, end) if i in excluded), end)
        excerpt = '\n'.join(lines[start:end])
        if len(excerpt) > 1400:
            continue
        vicinity = ' '.join(normalized[start:end])
        scores = [sum(3 if ' ' in word else 1 for word in words if word in vicinity) for words in groups]
        hits = {i for i, score in enumerate(scores) if score}
        direct = sum(term in vicinity for term in terms)
        preferred = int(bool(re.search(r'method|approach|framework|implementation|training|setting', sections[index])))
        candidates.append((start, end, hits, direct, excerpt, scores, preferred))
    selected, topic = [], 0
    while candidates and len(selected) < limit:
        eligible = [row for row in candidates if topic < len(groups) and row[5][topic]]
        best = max(eligible or candidates, key=lambda row: (
            row[5][topic] + row[6] * 2 if eligible else len(row[2]) + row[3], row[3], -row[0]))
        start, end, hits, direct, excerpt, scores, preferred = best
        selected.append({'source': filename, 'line_start': start + 1, 'line_end': end, 'text': excerpt})
        topic += 1
        candidates = [row for row in candidates if row[1] <= start or row[0] >= end]
    return selected


def whole_paper_passages(passages, budget):
    packed, omitted = [], 0
    for row in passages or []:
        if not isinstance(row, dict) or not isinstance(row.get('text'), str):
            continue
        candidate = {'passages': [*packed, row], 'omitted_due_to_budget': len(passages)}
        if len(json.dumps(candidate, ensure_ascii=False)) <= budget:
            packed.append(row)
        else:
            omitted += 1
    return {'passages': packed, 'omitted_due_to_budget': omitted,
            'scope': 'exact paper text, not code or execution evidence'}
