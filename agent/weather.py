import os
import asyncio
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi


@tool
def get_weather(city: str) -> str:
    """查询城市实时天气（Open-Meteo）"""
    url = f"https://wttr.in/{city}?format=j1&lang=zh"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        area_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]

        return (
            f"{area_name}({country})："
            f"温度{current['temp_C']}°C，"
            f"湿度{current['humidity']}%，"
            f"风速{current['windspeedKmph']} km/h，"
        )
    except Exception as e:
        print(f"❌ 查询失败：{e}")
        return "查询失败"


async def main():
    load_dotenv(override=True)
    qwen_api_key = os.getenv("DASHSCOPE_API_KEY")
    model = ChatTongyi(model="qwen-plus", api_key=qwen_api_key)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是中文天气助手。涉及实时天气必须调用工具，不要编造。",
    )

    inputs = {
        "messages": [
            {"role": "user", "content": "查北京和上海天气，并比较哪个更热。"}
        ]
    }

    print("===== 事件流 =====")
    async for event in agent.astream_events(inputs, version="v2"):
        et = event.get("event")
        name = event.get("name")
        data = event.get("data", {})

        if et == "on_tool_start":
            print(f"[TOOL START] {name} input={data.get('input')}")
        elif et == "on_tool_end":
            print(f"[TOOL END]   {name} output={data.get('output')}")
        elif et == "on_chat_model_stream":
            chunk = data.get("chunk")
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                print(content, end="", flush=True)

    print("\n===== 最终答案 =====")
    result = await agent.ainvoke(inputs)
    last_msg = result["messages"][-1]
    print(getattr(last_msg, "content", last_msg))


if __name__ == "__main__":
    asyncio.run(main())
