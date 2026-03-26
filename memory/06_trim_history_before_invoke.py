"""示例 6：历史裁剪（限制上下文长度）"""

import os
from dotenv import load_dotenv

from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in STORE:
        STORE[session_id] = InMemoryChatMessageHistory()
    return STORE[session_id]


def trim_history(inputs: dict) -> dict:
    # 仅保留最近 n 条历史消息，避免上下文无限增长
    print("DDD:", inputs)
    history = inputs.get("history", [])
    trimmed = trim_messages(
        history,
        token_counter=len,
        max_tokens=6,
        strategy="last",
        start_on="human",
    )
    inputs["history"] = trimmed
    return inputs


def main() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个简洁助手。"),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatTongyi(
        model="qwen-plus",
        max_retries=2,
        api_key=qwen_api_key,
    )
    chain = RunnableLambda(trim_history) | prompt | llm
    app = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "trim_demo"}}
    app.invoke({"input": "我叫小明"}, config=config)
    app.invoke({"input": "我喜欢羽毛球"}, config=config)
    app.invoke({"input": "我住在上海"}, config=config)
    result = app.invoke({"input": "总结一下我的信息"}, config=config)
    print(result.content)


if __name__ == "__main__":
    main()
