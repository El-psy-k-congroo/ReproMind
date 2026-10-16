"""Provider generation controls; reasoning never becomes the user answer.

GLM-5.3/Flash: https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3
Forced thinking requires a combined reasoning/answer budget and cannot be disabled.
"""
import re


def forced_thinking(model):
    return bool(re.match(r'^glm-5\.3(?:$|[-.])',model.lower().rsplit('/',1)[-1]))


def generation_options(model, requested_tokens, non_thinking, token_cap=8192):
    if forced_thinking(model):
        minimum = 4096 if non_thinking else 8192
        if token_cap < minimum:
            raise ValueError(f'{model} 始终启用推理；当前总输出预算 {token_cap} 过低，至少需要 {minimum}，或选择支持关闭推理的模型')
        return {'thinking':{'type':'enabled'},'reasoning_effort':'low' if non_thinking else 'high',
                'max_tokens':min(token_cap,max(minimum,requested_tokens+(2048 if non_thinking else 4096)))}
    if non_thinking and 'deepseek' in model.lower():
        return {'thinking':{'type':'disabled'},'reasoning_effort':'none','max_tokens':requested_tokens}
    return {'max_tokens':requested_tokens}
