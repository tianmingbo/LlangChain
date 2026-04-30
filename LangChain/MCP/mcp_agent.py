import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main() -> None:
    load_dotenv(override=True)

    qwen_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not qwen_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请先在 .env 中配置。")

    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")

    client = MultiServerMCPClient(
        {
            "mcp-http-tools": {
                "transport": "streamable_http",
                "url": mcp_server_url,
            }
        }
    )

    tools = await client.get_tools()
    model = ChatTongyi(model="qwen-plus", api_key=qwen_api_key)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个中文助手。遇到可由工具完成的计算或时间查询，优先调用工具。",
    )

    query = "计算 123 + 456。"
    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    print("用户问题:", query)
    print("Agent回答:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
