"""
================================================================================
LLM 调用封装 — llm.py
================================================================================

【费曼视角：一句话讲清楚】
llm_call() 是整个项目的"通讯中心"——所有与 LLM 的交互都经过这里。
底层用 LiteLLM（开源，12k+ stars，支持 100+ LLM）抹平不同 API 的差异。
配置从 llm_config.toml 读取，不硬编码任何 Key 或模型名。

就像你手机上的"电话"App——不管对方用的是移动、联通还是电信，
拨号界面都一样。LiteLLM 是通讯网络，llm_config.toml 是通讯录。

【配置文件】
    项目根目录 llm_config.toml（已加入 .gitignore，不会泄露 Key）

    [llm]
    model = "openai/deepseek-ai/DeepSeek-V4-Flash"
    api_key = "sk-xxx"
    api_base = "https://api.siliconflow.cn/v1"

【使用方式】
    # 用配置文件的默认设置
    llm_call("你好")

    # 临时覆盖 model
    llm_call("你好", model="gpt-4o")

【跨文件关系】
- 被 core.py 的 reason_from_future() 调用
- 被 core_nhx.py 的 reason_from_future_nhx() 调用
- 依赖 litellm（开源 LLM 统一接口库）
- 读取 llm_config.toml（项目根目录）
"""
import os
from pathlib import Path

import litellm


def _find_config_path() -> Path | None:
    """查找 llm_config.toml 配置文件。

    查找顺序：
    1. 当前工作目录
    2. 项目根目录（pyproject.toml 所在目录）
    """
    cwd = Path.cwd()
    candidate = cwd / "llm_config.toml"
    if candidate.exists():
        return candidate

    parent = cwd
    for _ in range(5):
        parent = parent.parent
        if (parent / "pyproject.toml").exists():
            candidate = parent / "llm_config.toml"
            if candidate.exists():
                return candidate
            break

    return None


def load_llm_config() -> dict:
    """从 llm_config.toml 加载 LLM 配置。

    返回 dict，包含 model, api_key, api_base 等字段。
    找不到配置文件则返回空 dict（回退到环境变量）。
    """
    config_path = _find_config_path()
    if config_path is None:
        return {}

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {}

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("llm", {})
    except Exception:
        return {}


_CONFIG = load_llm_config()

DEFAULT_MODEL = _CONFIG.get("model", os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-preview-05-20"))
DEFAULT_API_KEY = _CONFIG.get("api_key", "")
DEFAULT_API_BASE = _CONFIG.get("api_base", "")


def llm_call(
    prompt: str,
    *,
    model: str | None = None,
    verbose: bool = False,
    tools: list | None = None,
) -> str:
    """向 LLM 发送 prompt 并返回其文本回复。

    model 参数：
    - None → 使用 llm_config.toml 中配置的模型
    - "gpt-4o" → 临时切换到 OpenAI
    - "deepseek/deepseek-chat" → 临时切换到 DeepSeek
    - 完整列表：https://docs.litellm.ai/docs/providers
    """
    use_model = model or DEFAULT_MODEL

    if verbose:
        print(f"--- LLM PROMPT ({use_model}) ---")
        print(prompt)
        print("---------------------------")

    messages = [{"role": "user", "content": prompt}]

    kwargs: dict = {
        "model": use_model,
        "messages": messages,
    }

    if DEFAULT_API_KEY:
        kwargs["api_key"] = DEFAULT_API_KEY
    if DEFAULT_API_BASE:
        kwargs["api_base"] = DEFAULT_API_BASE

    if tools:
        kwargs["tools"] = tools

    response = litellm.completion(**kwargs)

    result_text = response.choices[0].message.content or ""

    if verbose:
        print(f"--- LLM RESPONSE ({use_model}) ---")
        print(result_text)
        print("----------------------------")

    return result_text
