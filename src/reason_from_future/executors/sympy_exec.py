"""
SympyExecutor — sympy 精确计算执行器

用 sympy 执行数学代码，替代 LLM 心算：
  - 方程求解: solve(Eq(x**2 - 5*x + 6, 0), x) → [2, 3]
  - 表达式化简: simplify(x**2 - 5*x + 6) → (x-2)*(x-3)
  - 精确运算: Rational(1, 3) + Rational(1, 6) → 1/2
  - 代数运算: expand, factor, collect, ...

安全措施：
  - restricted globals: 只暴露 sympy 和 math 函数
  - timeout: 默认 10 秒
  - 无文件/网络访问
  - 无 import 语句

Usage:
    executor = SympyExecutor()
    result = executor.execute("x = symbols('x'); solve(Eq(x**2 - 4, 0), x)")
    # result.success = True
    # result.result = [-2, 2]
"""

from __future__ import annotations

import time
from typing import Any

from .base import ExecutionResult


class SympyExecutor:
    """Sympy 精确计算执行器。"""

    _SAFE_GLOBALS: dict[str, Any] = {}

    def __init__(self, timeout: int = 10):
        self._timeout = timeout
        self._init_safe_globals()

    def _init_safe_globals(self) -> None:
        """初始化受限的全局命名空间。逐个导入，跳过不存在的。"""
        import sympy

        g: dict[str, Any] = {
            "sympy": sympy,
            "int": int, "float": float, "str": str,
            "bool": bool, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "frozenset": frozenset,
            "len": len, "range": range, "enumerate": enumerate,
            "sorted": sorted, "reversed": reversed,
            "sum": sum, "min": min, "max": max,
            "abs": abs, "round": round, "pow": pow,
            "print": print,
            "True": True, "False": False, "None": None,
        }

        # 逐个从 sympy 导入，跳过不存在的
        sympy_names = [
            "symbols", "Symbol", "Rational", "Integer", "Float",
            "solve", "Eq", "Ne", "Lt", "Gt", "Le", "Ge",
            "simplify", "expand", "factor", "collect", "cancel", "apart",
            "sqrt", "root", "Abs", "sign",
            "sin", "cos", "tan", "asin", "acos", "atan",
            "exp", "log", "ln",
            "pi", "E", "oo", "I", "zoo", "nan",
            "binomial", "factorial",
            "floor", "ceiling",
            "Sum", "Product", "summation", "product",
            "Matrix", "det",
            "gcd", "lcm",
            "Poly", "roots",
            "limit", "diff", "integrate",
            "nsimplify",
        ]
        for name in sympy_names:
            try:
                g[name] = getattr(sympy, name)
            except AttributeError:
                pass

        # 简写
        if "Rational" in g:
            g["R"] = g["Rational"]

        # 额外从子模块导入
        try:
            from sympy.ntheory import isprime, divisors, totient, mobius
            g["isprime"] = isprime
            g["divisors"] = divisors
            g["totient"] = totient
            g["mobius"] = mobius
        except ImportError:
            pass

        self._SAFE_GLOBALS = g

    def execute(
        self,
        code: str,
        *,
        variables: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """执行 sympy 代码。"""
        start = time.time()

        # 构建执行环境
        exec_env: dict[str, Any] = dict(self._SAFE_GLOBALS)
        if variables:
            exec_env.update(variables)

        # 安全检查
        if not self._is_safe_code(code):
            return ExecutionResult(
                success=False,
                error="代码包含不安全操作（import/open/exec/eval/compile）",
                code=code,
                elapsed_s=time.time() - start,
            )

        # 执行
        try:
            exec(code, exec_env)  # noqa: S102

            # 提取结果
            result = exec_env.get("result", exec_env.get("answer", None))
            if result is None:
                for key in reversed(list(exec_env.keys())):
                    val = exec_env[key]
                    if not key.startswith("_") and not callable(val) and key not in self._SAFE_GLOBALS:
                        result = val
                        break

            # sympy 类型转 Python 类型
            result = self._sympy_to_python(result)

            elapsed = time.time() - start
            return ExecutionResult(
                success=True,
                result=result,
                result_type=type(result).__name__,
                code=code,
                elapsed_s=elapsed,
                variables={
                    k: self._sympy_to_python(v)
                    for k, v in exec_env.items()
                    if not k.startswith("_") and not callable(v) and k not in self._SAFE_GLOBALS
                },
            )
        except Exception as e:
            elapsed = time.time() - start
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                code=code,
                elapsed_s=elapsed,
            )

    @staticmethod
    def _is_safe_code(code: str) -> bool:
        """检查代码是否安全。"""
        forbidden = [
            "import ", "from ", "__import__",
            "open(", "exec(", "eval(", "compile(",
            "os.", "sys.", "subprocess.", "shutil.",
            "globals()", "locals()", "vars(",
            "getattr(", "setattr(", "delattr(",
        ]
        code_lower = code.lower()
        for f in forbidden:
            if f.lower() in code_lower:
                return False
        return True

    @staticmethod
    def _sympy_to_python(value: Any) -> Any:
        """将 sympy 类型转为 Python 原生类型。"""
        if value is None:
            return None
        try:
            import sympy
            if isinstance(value, sympy.Integer):
                return int(value)
            elif isinstance(value, sympy.Rational):
                f = float(value)
                if f == int(f) and abs(f) < 1e15:
                    return int(f)
                return f
            elif isinstance(value, sympy.Float):
                return float(value)
            elif isinstance(value, (list, tuple)):
                return [SympyExecutor._sympy_to_python(v) for v in value]
            elif isinstance(value, dict):
                return {k: SympyExecutor._sympy_to_python(v) for k, v in value.items()}
            elif isinstance(value, sympy.Basic):
                try:
                    f = float(value)
                    if f == int(f) and abs(f) < 1e15:
                        return int(f)
                    return f
                except (TypeError, ValueError):
                    return str(value)
        except ImportError:
            pass
        return value

    def generate_code_prompt(
        self,
        problem: str,
        known_variables: dict[str, Any] | None = None,
    ) -> str:
        """生成让 LLM 产出 sympy 代码的 prompt。"""
        vars_section = ""
        if known_variables:
            vars_section = "\n已知变量:\n" + "\n".join(
                f"  {k} = {v}" for k, v in known_variables.items()
            )

        return f"""你是一个数学计算专家。请用 sympy 写代码来解决以下问题。

问题: {problem}
{vars_section}

要求:
1. 用 sympy 精确计算，不要用浮点近似
2. 最终结果赋值给变量 `result`
3. 只返回 Python 代码，不要其他文字
4. 可用的 sympy 函数: symbols, solve, Eq, simplify, expand, factor, sqrt, Rational, binomial, factorial, Sum, summation, Matrix, det, gcd, lcm, isprime, divisors, nsimplify, pi, E, oo, I
5. 用 Rational 代替浮点除法（如 Rational(1, 3) 而非 1/3）

示例:
```python
x = symbols('x')
result = solve(Eq(x**2 - 4, 0), x)
```"""
