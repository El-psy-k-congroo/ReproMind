"""Scope-aware reuse and dataset-specific paper parameters."""
import hashlib
import re


def specification_context(paper, goal):
    return {'paper_sha256':hashlib.sha256(paper.encode('utf-8')).hexdigest(),
            'goal':' '.join(goal.split())}


def reusable_specification(spec, paper, goal):
    return bool(spec and spec.get('spec_context') == specification_context(paper,goal))


def dataset_parameters(spec, dataset, *, scoped_only=False):
    common={};specific={}
    def key(value):return re.sub(r'[^a-z0-9]','',str(value).casefold())
    for row in (spec or {}).get('hyperparameters') or []:
        if not isinstance(row,dict):continue
        name=str(row.get('name') or '')
        if not name:continue
        labels=row.get('datasets') or ([row['dataset']] if row.get('dataset') else [])
        if isinstance(labels,str):labels=[labels]
        if not isinstance(labels,list):continue
        if labels and not any(key(label)==key(dataset) for label in labels):continue
        (specific if labels else common)[name]=row.get('value')
    return specific if scoped_only else {**common,**specific}
