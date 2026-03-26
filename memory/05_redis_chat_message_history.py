"""示例 5：RedisChatMessageHistory（Redis 持久化）"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://:tian666@localhost:6389/0",
        key_prefix="lc:memory:",
    )


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
    chain = prompt | llm
    app = RunnableWithMessageHistory(
        chain,
        get_redis_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "redis_user_1"}}
    app.invoke({"input": "我在北京工作"}, config=config)
    result = app.invoke({"input": "我在哪里工作？"}, config=config)
    print(result.content)


if __name__ == "__main__":
    main()
