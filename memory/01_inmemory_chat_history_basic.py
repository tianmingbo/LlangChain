"""示例 1：直接使用 InMemoryChatMessageHistory 存储多轮对话"""

from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


def main() -> None:
    llm = ChatTongyi(
        model="qwen-plus",
        max_retries=2,
        api_key=qwen_api_key,
    )
    # 创建内存记忆对象：仅保存在当前 Python 进程内
    history = InMemoryChatMessageHistory()

    # 第一轮：用户输入 -> 模型回答 -> 写入历史
    history.add_user_message("我叫小明")
    ai_msg_1 = llm.invoke(history.messages)
    history.add_message(ai_msg_1)
    print("第一轮:", ai_msg_1.content, "\n")

    # 第二轮：继续基于已有历史对话
    history.add_user_message("我叫什么？")
    ai_msg_2 = llm.invoke(history.messages)
    history.add_message(ai_msg_2)
    print("第二轮:", ai_msg_2.content, "\n")

    for msg in history.messages:
        print(msg.content)


if __name__ == "__main__":
    main()
