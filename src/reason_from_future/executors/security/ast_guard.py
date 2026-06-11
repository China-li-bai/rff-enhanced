"""AST 安全检查器 — 在代码执行前做语法树级安全验证

源自 Eis4TY/sym-mcp (MIT License)，适配 GRAVEC 架构。

相比字符串黑名单的优势：
  1. AST 级别检查，无法通过字符串拼接绕过
  2. 白名单语法节点，未知语法直接拒绝
  3. 精确到行号的错误报告
  4. 智能语法错误诊断（括号不匹配、缺少运算符等）

用法:
    from .ast_guard import validate_code, GuardResult
    result = validate_code(user_code)
    if not result.ok:
        print(result.message)  # 安全拦截原因
"""

from __future__ import annotations

import ast
import io
import keyword
import tokenize
from dataclasses import dataclass
import re


# 只允许导入 sympy 和 math
ALLOWED_MODULES = {"sympy", "math"}

# 禁止调用的内置函数
BLOCKED_NAMES = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "help",
    "dir",
    "type",
    "super",
}

# 禁止访问的根模块
BLOCKED_ROOT_NAMES = {
    "os", "sys", "subprocess", "pathlib", "socket",
    "importlib", "builtins",
}

# 允许的 AST 语法节点（白名单）
ALLOWED_NODES = {
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Call,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Return,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.Constant,
    ast.Import,
    ast.ImportFrom,
    ast.alias,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.keyword,
    ast.IfExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Attribute,
    ast.Try,
    ast.ExceptHandler,
    ast.Raise,
    ast.Assert,
    ast.Lambda,
    ast.NamedExpr,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.Delete,
    ast.With,
    ast.withitem,
    ast.Starred,
    # 运算符节点
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.MatMult, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
}


@dataclass(frozen=True)
class GuardResult:
    """安全检查结果。"""
    ok: bool
    message: str = ""


class SecurityViolation(ValueError):
    """代码未通过 AST 安全验证。"""


def validate_code(code: str) -> GuardResult:
    """验证代码是否安全可执行。

    Returns:
        GuardResult(ok=True) 如果代码安全
        GuardResult(ok=False, message=...) 如果代码不安全
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return GuardResult(ok=False, message=_format_syntax_error(code, exc))

    try:
        _AstValidator().visit(tree)
    except SecurityViolation as exc:
        return GuardResult(ok=False, message=str(exc))

    return GuardResult(ok=True)


class _AstValidator(ast.NodeVisitor):
    """AST 访问者，检查代码是否安全。"""

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in ALLOWED_NODES:
            raise SecurityViolation(
                f"安全拦截: 不允许的语法节点 `{type(node).__name__}`。"
            )
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_MODULES:
                allowed = ", ".join(sorted(ALLOWED_MODULES))
                raise SecurityViolation(
                    f"安全拦截: 第 {node.lineno} 行禁止导入模块 `{alias.name}`。"
                    f"当前沙箱仅允许导入: {allowed}。"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".")[0]
        if module not in ALLOWED_MODULES:
            allowed = ", ".join(sorted(ALLOWED_MODULES))
            raise SecurityViolation(
                f"安全拦截: 第 {node.lineno} 行禁止从模块 `{node.module}` 导入。"
                f"当前沙箱仅允许导入: {allowed}。"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        self._check_identifier(name, node.lineno)
        if name in BLOCKED_ROOT_NAMES:
            raise SecurityViolation(
                f"安全拦截: 第 {node.lineno} 行禁止访问 `{name}`。"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._check_identifier(node.attr, node.lineno)
        root = _root_name(node)
        if root in BLOCKED_ROOT_NAMES:
            raise SecurityViolation(
                f"安全拦截: 第 {node.lineno} 行禁止访问 `{root}`。"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _callable_name(node.func)
        if func_name and func_name in BLOCKED_NAMES:
            raise SecurityViolation(
                f"安全拦截: 第 {node.lineno} 行禁止调用 `{func_name}`。"
            )
        self.generic_visit(node)

    @staticmethod
    def _check_identifier(name: str, lineno: int) -> None:
        if "__" in name:
            raise SecurityViolation(
                f"安全拦截: 第 {lineno} 行出现双下划线标识符。"
            )


# ============================================================================
# 辅助函数
# ============================================================================

def _root_name(node: ast.AST) -> str | None:
    """获取属性链的根名称，如 os.path.join → os。"""
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _callable_name(node: ast.AST) -> str | None:
    """获取可调用对象的名称。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _format_syntax_error(code: str, exc: SyntaxError) -> str:
    """格式化语法错误，附带智能诊断。"""
    lineno = exc.lineno or 1
    offset = exc.offset or 1
    source_line = _get_source_line(code, lineno)
    parts = [f"语法错误: 第 {lineno} 行, 第 {offset} 列, {exc.msg}"]
    if source_line:
        parts.append(f"源码: {source_line.strip()}")
        parts.append(f"位置: {' ' * max(offset - 1, 0)}^")
    reasons = _diagnose_syntax_error(source_line, exc.msg, offset)
    if reasons:
        parts.append("可能原因: " + "；".join(reasons) + "。")
    blocked_imports = _scan_blocked_imports(code)
    if blocked_imports:
        parts.append(
            "另外检测到不支持的导入: "
            + ", ".join(blocked_imports)
            + "；当前沙箱仅允许导入: "
            + ", ".join(sorted(ALLOWED_MODULES))
            + "。"
        )
    return " ".join(parts)


def _get_source_line(code: str, lineno: int) -> str:
    lines = code.splitlines()
    if lineno < 1 or lineno > len(lines):
        return ""
    return lines[lineno - 1]


def _diagnose_syntax_error(
    source_line: str, message: str, offset: int
) -> list[str]:
    """智能诊断语法错误原因。"""
    reasons: list[str] = []
    tokens = _line_tokens(source_line)

    if "was never closed" in message or "unexpected EOF" in message:
        reasons.append("括号、方括号或花括号没有闭合")
    if "unterminated string literal" in message:
        reasons.append("字符串字面量没有闭合；请检查引号是否成对出现")
    if "invalid decimal literal" in message:
        reasons.append("数字字面量格式不合法；数字和变量名之间需要运算符")
    if "expected ':'" in message:
        reasons.append("控制语句或函数定义末尾缺少冒号 `:`")
    if "expected an indented block" in message:
        reasons.append("代码块缺少缩进内容")

    reasons.extend(_diagnose_adjacent_expressions(tokens))
    reasons.extend(_diagnose_operator_operands(tokens, offset))

    return _dedupe(reasons)


def _line_tokens(source_line: str) -> list[tokenize.TokenInfo]:
    try:
        return [
            tok
            for tok in tokenize.generate_tokens(
                io.StringIO(source_line).readline
            )
            if tok.type
            not in {
                tokenize.ENCODING,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.ENDMARKER,
            }
        ]
    except tokenize.TokenError:
        return []


def _diagnose_adjacent_expressions(
    tokens: list[tokenize.TokenInfo],
) -> list[str]:
    reasons: list[str] = []
    for prev, cur in zip(tokens, tokens[1:]):
        if _is_expr_end(prev) and _is_expr_start(cur):
            if prev.string == ")" and cur.type == tokenize.NUMBER:
                reasons.append(
                    "右括号后直接跟数字不是合法写法；"
                    "如需幂运算请使用 `**`，如需乘法请使用 `*`"
                )
            else:
                reasons.append(
                    "两个表达式相邻但中间缺少运算符，"
                    "例如 `+`、`-`、`*`、`/` 或 `**`"
                )
    return reasons


def _diagnose_operator_operands(
    tokens: list[tokenize.TokenInfo], offset: int
) -> list[str]:
    reasons: list[str] = []
    binary_ops = {
        "+", "-", "*", "/", "//", "%", "**", "@",
        "==", "!=", "<", "<=", ">", ">=",
        "=", "+=", "-=", "*=", "/=",
    }
    expr_boundary = {
        "(", "[", "{", ",", ":",
        "=", "+=", "-=", "*=", "/=",
        "==", "!=", "<", "<=", ">", ">=",
    }

    for idx, tok in enumerate(tokens):
        if tok.type != tokenize.OP or tok.string not in binary_ops:
            continue
        prev = _previous_significant(tokens, idx)
        next_tok = _next_significant(tokens, idx)
        if tok.string not in {"+", "-"} and (
            prev is None
            or prev.string in expr_boundary
            or prev.string in binary_ops
        ):
            reasons.append(
                f"运算符 `{tok.string}` 前缺少左操作数"
            )
        if next_tok is None or (
            next_tok.string in {")", "]", "}", ",", ":"}
            or next_tok.string in binary_ops
        ):
            reasons.append(
                f"运算符 `{tok.string}` 后缺少右操作数"
            )

    if not reasons:
        near = _token_near_offset(tokens, offset)
        if near and near.type == tokenize.OP and near.string in binary_ops:
            reasons.append(
                f"请检查运算符 `{near.string}` 两侧是否都有合法表达式"
            )
    return reasons


def _previous_significant(
    tokens: list[tokenize.TokenInfo], idx: int
) -> tokenize.TokenInfo | None:
    if idx <= 0:
        return None
    return tokens[idx - 1]


def _next_significant(
    tokens: list[tokenize.TokenInfo], idx: int
) -> tokenize.TokenInfo | None:
    if idx + 1 >= len(tokens):
        return None
    return tokens[idx + 1]


def _token_near_offset(
    tokens: list[tokenize.TokenInfo], offset: int
) -> tokenize.TokenInfo | None:
    column = max(offset - 1, 0)
    for tok in tokens:
        if tok.start[1] <= column <= tok.end[1]:
            return tok
    for tok in tokens:
        if tok.start[1] >= column:
            return tok
    return tokens[-1] if tokens else None


def _is_expr_end(tok: tokenize.TokenInfo) -> bool:
    return (
        (
            tok.type == tokenize.NAME
            and not keyword.iskeyword(tok.string)
        )
        or tok.type in {tokenize.NUMBER, tokenize.STRING}
        or tok.string in {")", "]", "}"}
    )


def _is_expr_start(tok: tokenize.TokenInfo) -> bool:
    return (
        (
            tok.type == tokenize.NAME
            and not keyword.iskeyword(tok.string)
        )
        or tok.type in {tokenize.NUMBER, tokenize.STRING}
        or tok.string in {"(", "[", "{"}
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _scan_blocked_imports(code: str) -> list[str]:
    blocked: list[str] = []
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        import_match = re.match(r"import\s+([A-Za-z_][\w.]*)", line)
        from_match = re.match(
            r"from\s+([A-Za-z_][\w.]*)\s+import\b", line
        )
        module = ""
        if import_match:
            module = import_match.group(1)
        elif from_match:
            module = from_match.group(1)
        if (
            module
            and module.split(".")[0] not in ALLOWED_MODULES
            and module not in blocked
        ):
            blocked.append(module)
    return blocked
