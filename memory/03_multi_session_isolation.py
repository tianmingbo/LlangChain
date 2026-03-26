"""示例 3：多会话隔离（不同 session_id 的记忆互不影响）"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# 这里用 dict 模拟会话存储，键是 session_id
HISTORY_STORE: dict[str, InMemoryChatMessageHistory] = {}

STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in STORE:
        STORE[session_id] = InMemoryChatMessageHistory()
    return STORE[session_id]


def main() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个助手。"),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatTongyi(
        model="qwen-plus",
        max_retries=2,
        api_key=qwen_api_key,
    )
    app = RunnableWithMessageHistory(
        prompt | llm,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config_a = {"configurable": {"session_id": "user_A"}}
    config_b = {"configurable": {"session_id": "user_B"}}

    app.invoke({"input": "我叫小明"}, config=config_a)
    app.invoke({"input": "我叫小红"}, config=config_b)

    ans_a = app.invoke({"input": "我叫什么？"}, config=config_a)
    ans_b = app.invoke({"input": "我叫什么？"}, config=config_b)

    print("A:", ans_a.content)
    print("B:", ans_b.content)


if __name__ == "__main__":
    main()
