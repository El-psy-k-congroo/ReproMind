"""Classify the terminal exception, not an optional import earlier in its chain."""
import re


def terminal_exception(output):
    # Some libraries use names such as NVMLError_LibraryNotFound. The final
    # exception need not end with "Error"; otherwise an earlier chained error
    # is mistaken for the cause of the process exit.
    matches = list(re.finditer(r'(?m)^([A-Za-z_][\w.]*(?:Error|Exception)[\w.]*)[ \t]*:[ \t]*(.*)$',output))
    return (matches[-1][1],matches[-1][2]) if matches else ('','')


def compatibility_plan(output):
    kind,detail = terminal_exception(output)
    if (kind == 'TypeError' and 'Descriptors cannot be created directly' in detail
            and '_pb2.py' in output and 'protobuf package to 3.20.x' in output):
        matches = re.findall(r'site-packages[\\/]([^"\n]+_pb2)\.py',output)
        module = matches[-1].replace('\\','.').replace('/','.') if matches else 'google.protobuf'
        if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*',module): module = 'google.protobuf'
        return {'package':'protobuf','version':'3.20.3','import_probe':module,
                'reason':'最终异常确认旧生成 protobuf 接口不兼容；按异常给出的 3.20.x 兼容范围处理，不安装前面可选导入的 TensorFlow。',
                'reference':'https://protobuf.dev/news/2022-05-06/#python-updates'}
    return None
