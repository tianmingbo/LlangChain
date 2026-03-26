"""示例 4：FileChatMessageHistory（本地文件持久化）"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")


def get_file_history(session_id: str) -> BaseChatMessageHistory:
    # 每个会话一个文件，程序重启后历史仍可恢复
    history_dir = Path(".history")
    history_dir.mkdir(parents=True, exist_ok=True)
    file_path = history_dir / f"{session_id}.json"
    return FileChatMessageHistory(str(file_path), encoding="utf-8")


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
        get_file_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "file_demo"}}
    app.invoke({"input": "我喜欢篮球"}, config=config)
    result = app.invoke({"input": "我喜欢什么运动？"}, config=config)
    print(result.content)


if __name__ == "__main__":
    main()
