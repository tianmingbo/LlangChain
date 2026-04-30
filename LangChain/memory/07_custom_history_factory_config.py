"""示例 7：自定义 history_factory_config（多维度会话键）"""
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_history_by_user_conversation(user_id: str, conversation_id: str) -> InMemoryChatMessageHistory:
    # 用 user_id + conversation_id 组成唯一会话键
    key = f"{user_id}:{conversation_id}"
    if key not in STORE:
        STORE[key] = InMemoryChatMessageHistory()
    return STORE[key]


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
        get_history_by_user_conversation,
        input_messages_key="input",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(id="user_id", annotation=str, name="User ID", description="用户 ID"),
            ConfigurableFieldSpec(id="conversation_id", annotation=str, name="Conversation ID",
                                  description="会话 ID", ),
        ],
    )

    config = {"configurable": {"user_id": "u001", "conversation_id": "c001"}}
    app.invoke({"input": "我叫小明"}, config=config)
    result = app.invoke({"input": "我叫什么？"}, config=config)
    print(result.content)


if __name__ == "__main__":
    main()
