import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


async def main() -> None:
    async with streamable_http_client(MCP_SERVER_URL) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", [tool.name for tool in tools.tools])

            result = await session.call_tool("add", arguments={"a": 3, "b": 5})
            print("tool result:", result.structuredContent or result.content)

            resource = await session.read_resource("note://http-demo")
            print("resource:", resource.contents[0].text)

            prompt = await session.get_prompt(
                "weather_brief_prompt",
                arguments={"city": "Shanghai", "date": "2026-04-28"},
            )
            print("prompt:", prompt.messages[0].content.text)


if __name__ == "__main__":
    asyncio.run(main())
