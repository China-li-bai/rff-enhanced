"""结构化错误解析器 — 将原始异常转为标准错误码 + 修复提示

源自 Eis4TY/sym-mcp (MIT License)，适配 GRAVEC 架构。

错误码体系：
  E_AST_BLOCK  — AST 安全检查拦截（不安全代码）
  E_SYNTAX     — 语法错误
  E_TIMEOUT    — 执行超时
  E_MEMORY     — 内存不足
  E_RUNTIME    — 运行时错误（零除、未定义变量等）
  E_INTERNAL   — 内部错误

用法:
    from .error_parser import parse_error, ParsedError
    parsed = parse_error(traceback_text)
    print(parsed.code)   # "E_RUNTIME"
    print(parsed.hint)   # "运行时错误。请根据行号检查..."
"""

from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_HINT_LEVEL = "medium"


@dataclass(frozen=True)
class ParsedError:
    """结构化错误信息。"""
    code: str       # 错误码（E_AST_BLOCK / E_SYNTAX / E_TIMEOUT / E_MEMORY / E_RUNTIME / E_INTERNAL）
    line: int | None  # 出错行号
    err: str        # 精简的错误描述
    hint: str       # 修复提示


def parse_guard_result(message: str, hint_level: str = DEFAULT_HINT_LEVEL) -> ParsedError:
    """解析 AST 安全检查的错误信息。"""
    if message.startswith("语法错误"):
        code = "E_SYNTAX"
    else:
        code = "E_AST_BLOCK"
    line = _extract_line_from_message(message)
    return ParsedError(
        code=code, line=line, err=message,
        hint=build_hint(code, hint_level),
    )


def parse_runtime_error(
    traceback_text: str,
    hint_level: str = DEFAULT_HINT_LEVEL,
) -> ParsedError:
    """解析运行时异常的 traceback。"""
    line = _extract_user_line(traceback_text)
    err_text = _extract_error_text(traceback_text)
    code = _classify_error(err_text)
    if code == "E_RUNTIME":
        err_text = _format_runtime_error(traceback_text, err_text)
    return ParsedError(
        code=code, line=line, err=err_text,
        hint=build_hint(code, hint_level),
    )


def parse_timeout_error(hint_level: str = DEFAULT_HINT_LEVEL) -> ParsedError:
    """构造超时错误。"""
    return ParsedError(
        code="E_TIMEOUT", line=None,
        err="TimeoutError: 计算超时",
        hint=build_hint("E_TIMEOUT", hint_level),
    )


def parse_internal_error(
    message: str,
    hint_level: str = DEFAULT_HINT_LEVEL,
) -> ParsedError:
    """构造内部错误。"""
    clean = (message or "").strip() or "internal error"
    return ParsedError(
        code="E_INTERNAL", line=None, err=clean,
        hint=build_hint("E_INTERNAL", hint_level),
    )


def build_hint(code: str, hint_level: str = DEFAULT_HINT_LEVEL) -> str:
    """根据错误码生成修复提示。"""
    if hint_level == "none":
        return ""
    if hint_level == "short":
        return "根据错误码与行号最小改动后重试。"

    hints = {
        "E_AST_BLOCK": (
            "检测到不安全语句。"
            "请仅保留 sympy/math 相关计算代码，并移除系统调用后重试。"
        ),
        "E_SYNTAX": (
            "代码存在语法错误。"
            "请先修正报错行附近的括号、缩进或符号，再重新执行。"
        ),
        "E_TIMEOUT": (
            "计算超时。"
            "请减少计算规模、拆分步骤或先做代数化简后再求解。"
        ),
        "E_MEMORY": (
            "内存不足。"
            "请降低矩阵维度/展开规模，避免一次性构造超大对象。"
        ),
        "E_RUNTIME": (
            "运行时错误。"
            "请根据行号检查变量类型、零除、未定义变量等问题后重试。"
        ),
        "E_INTERNAL": (
            "服务内部异常。"
            "请重试一次；若仍失败请保留输入代码用于排查。"
        ),
    }
    return hints.get(code, hints["E_RUNTIME"])


# ============================================================================
# 内部辅助函数
# ============================================================================

def _extract_user_line(tb_text: str) -> int | None:
    frame = _extract_last_user_frame(tb_text)
    if frame is None:
        return None
    return frame[0]


def _extract_error_text(tb_text: str) -> str:
    default = "RuntimeError: 未知错误"
    if not tb_text.strip():
        return default
    tail = tb_text.strip().splitlines()[-1].strip()
    if ":" not in tail and re.fullmatch(r"[A-Za-z_]\w*", tail):
        return f"{tail}: 未知错误"
    if ":" not in tail:
        return default
    etype, msg = tail.split(":", 1)
    etype = etype.strip() or "RuntimeError"
    msg = msg.strip() or "未知错误"
    return f"{etype}: {msg}"


def _format_runtime_error(tb_text: str, err_text: str) -> str:
    parts = [err_text]
    source_line = _extract_user_source_line(tb_text)
    if source_line:
        parts.append(f"源码: {source_line.strip()}")
    reasons = _diagnose_runtime_error(err_text, source_line)
    if reasons:
        parts.append("可能原因: " + "；".join(reasons) + "。")
    return " ".join(parts)


def _extract_user_source_line(tb_text: str) -> str:
    frame = _extract_last_user_frame(tb_text)
    if frame is None:
        return ""
    return frame[1]


def _extract_last_user_frame(
    tb_text: str,
) -> tuple[int, str] | None:
    if not tb_text.strip():
        return None
    lines = tb_text.rstrip().splitlines()
    last: tuple[int, str] | None = None
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if not s.startswith('File "<user_code>", line '):
            continue
        try:
            line_no = int(s.split("line ")[1].split(",")[0])
        except (IndexError, ValueError):
            continue
        source_line = ""
        if idx + 1 < len(lines):
            candidate = lines[idx + 1].strip()
            if not candidate.startswith("File ") and not re.match(
                r"[A-Za-z_]\w*(Error|Exception)?:", candidate
            ):
                source_line = lines[idx + 1]
        last = (line_no, source_line)
    return last


def _diagnose_runtime_error(
    err_text: str, source_line: str
) -> list[str]:
    etype, _, msg = err_text.partition(":")
    reasons: list[str] = []
    if etype == "TypeError" and "object is not callable" in msg:
        reasons.append(
            "不可调用对象被当作函数调用；请检查括号前面的对象是否真的是函数"
        )
        if "Symbol" in msg:
            reasons.append(
                "SymPy 的 `symbols(...)` 会创建符号而不是函数；"
                "数学函数请使用 `sympy` 中的函数，例如 `log(...)`、`sin(...)`"
            )
    if etype == "NameError" and "is not defined" in msg:
        reasons.append(
            "变量或函数未定义；请检查名称拼写、定义顺序或是否被重新赋值覆盖"
        )
    if etype == "AttributeError":
        reasons.append(
            "对象没有被访问的属性或方法；请检查对象类型是否符合预期"
        )
    if etype == "TypeError" and "unsupported operand type" in msg:
        reasons.append(
            "运算符两侧对象类型不兼容；"
            "请检查数值、符号表达式和字符串是否混用"
        )
    if etype == "ZeroDivisionError":
        reasons.append("出现除以零；请检查分母或极限点附近的表达式")
    if source_line and re.search(r"symbols\s*\([^)]*\)\s*\(", source_line):
        reasons.append(
            "检测到 `symbols(...)(...)` 形式，这通常是把符号误当函数调用"
        )
    return _dedupe(reasons)


def _classify_error(err_text: str) -> str:
    etype = err_text.split(":", 1)[0]
    if etype in {"MemoryError"}:
        return "E_MEMORY"
    if etype in {"TimeoutError"}:
        return "E_TIMEOUT"
    return "E_RUNTIME"


def _extract_line_from_message(message: str) -> int | None:
    m = re.search(r"第\s*(\d+)\s*行", message)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
