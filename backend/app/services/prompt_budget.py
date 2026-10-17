"""Pack complete evidence records; never cut a formula or source line in half."""
import json


def whole_source_packets(evidence, budget):
    if not isinstance(evidence, dict):
        return evidence
    order = ['selected_entry_calls_not_execution_proof','selected_factory_calls_not_execution_proof','named_implementations', 'question_calculations_not_execution_proof',
             'author_configs', 'explicitly_named_functions', 'loss_assignments',
             'optimizer_assignments', 'static_model_attributes_not_runtime_shapes']
    packed = {key: value for key, value in evidence.items() if not isinstance(value, list)}
    lists = [key for key in order if isinstance(evidence.get(key), list)]
    lists.extend(key for key, value in evidence.items() if isinstance(value, list) and key not in lists)
    counts = {}
    for index in range(max((len(evidence[key]) for key in lists), default=0)):
        for key in lists:
            if index >= len(evidence[key]):
                continue
            row = evidence[key][index]
            candidate = {**packed, key: [*packed.get(key, []), row]}
            if len(json.dumps(candidate, ensure_ascii=False, default=str)) <= budget:
                packed = candidate
            else:
                counts[key] = counts.get(key, 0) + 1
    if counts:
        packed['omitted_packets_due_to_budget_not_missing'] = counts
    return packed
