"""
================================================================================
LLM 调用封装 — llm.py
================================================================================

【费曼视角：一句话讲清楚】
把"给 Gemini 发问题、收回答"这个操作打包成一个函数。
就像你用 ChatGPT 的输入框——打字进去，看回复出来。这里的 llm_call 就是那个输入框。

【Python 入门知识】
1. os.getenv("KEY")：从操作系统的环境变量中读取配置。
   比喻：你告诉朋友"你去我家，钥匙在门口地毯下面"——环境变量就是那串钥匙。
   用法：先在终端执行 `export GEMINI_API_KEY="your_key_here"`，然后 Python 就能读到。
   
2. 模块级代码（不在函数里的代码）在 import 时执行一次。这里的 client 创建就是如此。
   
3. types.Content / types.Part：Google GenAI SDK 的数据结构。
   就像寄快递：prompt 是"货物"，Part 是"包装"，Content 是"快递盒"。

【跨文件关系】
- 被 core.py 的 reason_from_future() 主循环调用
- 被 specs 中的 prompt 构建方法间接使用（通过 core.py 中转）
- 依赖环境变量 GEMINI_API_KEY
"""
import os
from google import genai
from google.genai import types

# [Python知识] os.getenv() 从环境变量读值，不存在则返回 None
# 必须先在终端执行：export GEMINI_API_KEY="your_actual_key"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# [Python知识] 模块级初始化：import 这个文件时就创建 client 对象
# 优点：只创建一次，不用每次调用 llm_call 都重新创建
# 注意：如果 API Key 是 None，这里不会报错，但后续调用会出错
client = genai.Client(api_key=GEMINI_API_KEY)


def llm_call(
    prompt: str,          # 要发给 LLM 的提示词文本
    *,                    # [Python知识] 后面的参数必须用关键字传递
    model: str = "gemini-2.5-flash-preview-05-20",  # LLM 模型名
    verbose: bool = False,  # 是否打印完整的 prompt 和 response（调试用）
    tools: list | None = None,  # [跨文件] 给 LLM 用的工具定义，来自 tools/ 目录
) -> str:
    """向 Gemini LLM 发送 prompt 并返回其文本回复。
    
    [费曼解释] 这个函数做什么：
    1. 收下你的问题（prompt）
    2. 包装成 Gemini 能理解的格式（types.Content）
    3. 发送给 Google 的服务器
    4. 等回复
    5. 把回复的文本取出来返回给你
    
    就这么简单！就像发微信消息——写内容（prompt），点发送（API 调用），等回复。
    """

    # 安全检查：没有 API Key 就报错（这也是为什么必须先设环境变量）
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set. "
            "Run: export GEMINI_API_KEY='your_key_here'"
        )

    if verbose:
        print(f"--- LLM PROMPT ({model}) ---")
        print(prompt)
        print("---------------------------")

    # [Python知识] 构建 SDK 要求的消息格式
    # types.Content 封装了一个完整的消息（包含角色和内容）
    # types.Part 封装了一段内容片段（可以是文本、图片等）
    # 这里只用了文本片段
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    # [跨文件] 如果有 tools（工具），就配置函数调用
    # tools 参数来自 CodeWritingWithToolsSpec，包含了 read_file / write_to_file 的 schema
    cfg = None
    if tools:
        cfg = types.GenerateContentConfig(tools=tools)

    # 真正的 API 调用：发送消息给 Gemini，阻塞等待回复
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=cfg,
    )

    # [Python知识] response.text 是 Gemini 返回的纯文本内容
    # 如果 Gemini 用了 function calling（工具调用），
    # SDK 会自动处理，最终 response.text 是处理后的结果
    result_text = response.text

    if verbose:
        print(f"--- LLM RESPONSE ({model}) ---")
        print(result_text)
        print("----------------------------")

    return result_text