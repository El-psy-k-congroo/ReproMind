"""Non-executing checks and evidence contracts for generated Python projects.

These checks never import generated modules. They are not a sandbox or proof of
runtime/scientific correctness; executable validation belongs to approved runs.
"""

import ast
import builtins
import hashlib
import inspect
import symtable


def dynamic_namespace(source: str) -> bool:
    return any(isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
               or isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id in {"exec", "globals"} for node in ast.walk(ast.parse(source)))


def undefined_globals(name: str, source: str) -> list[str]:
    table = symtable.symtable(source, name, "exec")
    # Dynamic namespaces cannot be soundly resolved by this small static check.
    if dynamic_namespace(source):
        return []
    known = set(dir(builtins)) | {
        "__name__", "__file__", "__package__", "__doc__", "__spec__",
        "__loader__", "__builtins__", "__cached__", "__annotations__",
    }
    known.update(symbol.get_name() for symbol in table.get_symbols()
                 if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace())
    missing = set()

    def inspect(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:
                missing.add(symbol.get_name())
        for child in scope.get_children():
            inspect(child)

    inspect(table)
    return sorted(missing)


def static_risks(files: dict[str, str]) -> list[dict]:
    """Hints for the reviewer, not automatic findings of algorithmic failure."""
    risks = []
    for name, source in files.items():
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.While) and any(isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name) and child.func.id == "len"
                    for child in ast.walk(node.test)):
                risks.append({"file": name, "line": node.lineno,
                              "code_evidence": lines[node.lineno - 1],
                              "reason": "检查按有效样本数量终止的循环：连续生成失败时，是否有独立的尝试次数上限和明确错误出口。内层循环上限不等于外层上限。"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dropna":
                risks.append({"file": name, "line": node.lineno,
                              "code_evidence": lines[node.lineno - 1],
                              "reason": "核对丢弃无效行后是否记录数量，并按实验要求补足样本或明确报告不足；此提示本身不证明存在错误。"})
    return risks[:20]


def local_call_errors(files: dict[str, str]) -> dict[str, str]:
    """Bind statically known local function calls, without importing modules.

    Dynamic calls, shadowed names, comprehensions and *args/**kwargs are left
    to runtime validation rather than producing speculative blockers.
    """
    trees = {name: ast.parse(source) for name, source in files.items()}

    def signature(node):
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        default_start = len(positional) - len(args.defaults)
        params = [inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_ONLY
                  if index < len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                  default=None if index >= default_start else inspect.Parameter.empty)
                  for index, arg in enumerate(positional)]
        if args.vararg:
            params.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
        params.extend(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                      default=inspect.Parameter.empty if default is None else None)
                      for arg, default in zip(args.kwonlyargs, args.kw_defaults))
        if args.kwarg:
            params.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
        return inspect.Signature(params)

    exports = {name.removesuffix('.py'): {node.name: signature(node) for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.decorator_list}
               for name, tree in trees.items()}
    errors = {}
    for name, tree in trees.items():
        if dynamic_namespace(files[name]):
            continue
        api = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.decorator_list:
                    api[node.name] = signature(node)
                else:
                    api.pop(node.name, None)
            elif isinstance(node, ast.ImportFrom) and node.module in exports:
                for alias in node.names:
                    if alias.name in exports[node.module]:
                        api[alias.asname or alias.name] = exports[node.module][alias.name]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(target, ast.Name):
                        api.pop(target.id, None)
        table = symtable.symtable(files[name], name, 'exec')

        class Calls(ast.NodeVisitor):
            scope = table

            def visit_FunctionDef(self, node):
                # Defaults/decorators execute in the enclosing scope.
                for value in [*node.decorator_list, *node.args.defaults,
                              *(x for x in node.args.kw_defaults if x is not None)]:
                    self.visit(value)
                outer = self.scope
                inner = next((child for child in outer.get_children()
                              if child.get_name() == node.name and child.get_lineno() == node.lineno), None)
                if inner is None:
                    return
                self.scope = inner
                for statement in node.body:
                    self.visit(statement)
                self.scope = outer

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                outer = self.scope
                inner = next((child for child in outer.get_children()
                              if child.get_name() == node.name and child.get_lineno() == node.lineno), None)
                if inner:
                    self.scope = inner
                    for statement in node.body:
                        self.visit(statement)
                    self.scope = outer

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in api:
                    symbol = self.scope.lookup(node.func.id)
                    if (self.scope.get_type() == 'module' or symbol.is_global()) and not (
                            any(isinstance(x, ast.Starred) for x in node.args)
                            or any(x.arg is None for x in node.keywords)):
                        try:
                            api[node.func.id].bind(*[None for _ in node.args], **{x.arg: None for x in node.keywords})
                        except TypeError as exc:
                            errors[name] = f"第 {node.lineno} 行调用本地接口 {node.func.id} 参数不匹配：{exc}"
                self.generic_visit(node)

            # Their symbol scopes are intentionally not approximated.
            def visit_Lambda(self, node):
                return

            visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

        Calls().visit(tree)
    return errors


def grounded_issue(issue: dict, files: dict[str, str], paper: str) -> dict:
    """Reject invented files/quotes; do not silently redirect a defect to main."""
    name = issue.get("file")
    quote = issue.get("code_evidence")
    reason = issue.get("reason")
    category = issue.get("category", "code")
    if not isinstance(name, str) or name not in files or not isinstance(reason, str) or not reason.strip():
        raise ValueError("审查意见缺少有效文件名或原因")
    if not isinstance(category, str) or category not in {"code", "runtime", "interface", "data", "method"}:
        raise ValueError(f"{name}：问题 category 必须是 code/runtime/interface/data/method")
    if not isinstance(quote, str) or not quote.strip() or quote not in files[name]:
        raise ValueError(f"{name}：审查意见缺少可定位的原文源码证据")
    if len(quote) > 2000 or len(quote.splitlines()) > 25 or files[name].count(quote) != 1:
        raise ValueError(f"{name}：源码证据必须是唯一可定位的短片段（最多 25 行 / 2000 字符）")
    paper_quote = issue.get("paper_evidence", "")
    if category == "method" and (not isinstance(paper_quote, str)
                                  or not paper_quote.strip() or paper_quote not in paper):
        raise ValueError(f"{name}：方法偏差缺少论文原文证据，应列为待核实说明")
    identity = hashlib.sha256(f"{name}\0{category}\0{quote}".encode()).hexdigest()[:16]
    return {"id": identity, "file": name, "reason": reason.strip()[:1200],
            "category": str(category), "code_evidence": quote,
            "paper_evidence": paper_quote, "status": "open"}


def unrelated_definition_error(original: str, candidate: str, quotes: list[str]) -> str:
    """Protect existing top-level definitions outside the evidenced repair scope."""
    old_tree, new_tree = ast.parse(original), ast.parse(candidate)
    touched = set()
    for quote in quotes:
        start = original.find(quote)
        if start < 0:
            continue
        first = original.count("\n", 0, start) + 1
        last = first + quote.count("\n")
        touched.update(range(first, last + 1))
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    new_nodes = {(type(node), node.name): node for node in new_tree.body if isinstance(node, definitions)}
    def inspect(nodes, candidates):
        for node in nodes:
            if not isinstance(node, definitions):
                continue
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            replacement = candidates.get((type(node), node.name))
            if not touched.intersection(range(start, (node.end_lineno or node.lineno) + 1)):
                if replacement is None or ast.dump(node, include_attributes=False) != ast.dump(replacement, include_attributes=False):
                    return f"修改了证据范围之外的定义 {node.name}；请保留无关函数/类，只修复已定位问题"
            elif isinstance(node, ast.ClassDef) and isinstance(replacement, ast.ClassDef):
                # A quote in forward() must not authorize rewriting __init__ or
                # unrelated methods in the same class.
                error = inspect(node.body, {(type(child), child.name): child for child in replacement.body
                                            if isinstance(child, definitions)})
                if error:
                    return error
        return ""

    return inspect(old_tree.body, new_nodes)
