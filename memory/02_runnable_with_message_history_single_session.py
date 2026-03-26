"""示例 2：RunnableWithMessageHistory（单会话），支持LCEL语法链式调用"""

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


# 获取会话历史
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in HISTORY_STORE:
        HISTORY_STORE[session_id] = InMemoryChatMessageHistory()
    return HISTORY_STORE[session_id]


def main() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个简洁的助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatTongyi(
        model="qwen-plus",
        max_retries=2,
        api_key=qwen_api_key,
    )
    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 记忆获取函数
        input_messages_key="input",  # 输入消息的键
        history_messages_key="history",  # 历史消息的键，和prompt对应
    )

    config = {"configurable": {"session_id": "user_001"}}  # config必须这么写。只有 session_id 才能生效

    chain_with_history.invoke({"input": "我叫小明"}, config=config)
    result = chain_with_history.invoke({"input": "我叫什么？"}, config=config)
    print(result.content)


if __name__ == "__main__":
    main()
