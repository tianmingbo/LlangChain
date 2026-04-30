import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=qwen_api_key,
    model="qwen-turbo",
    temperature=0.1
)


# ----------------------
# 2. 定义多个工具
# ----------------------

# 工具1：天气查询
class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，如：北京")


@tool("get_weather", args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    print(f"【工具调用】执行天气查询：{city}")
    return f"{city}：晴天，25℃，微风"


# 工具2：计算器
class CalcInput(BaseModel):
    a: int = Field(description="数字1")
    b: int = Field(description="数字2")


@tool("calculator", args_schema=CalcInput)
def calculator(a: int, b: int) -> int:
    """简单的两个数加法计算器"""
    print(f"【工具调用】执行计算：{a} + {b}")
    return a + b


# 工具3：保存文件
class FileInput(BaseModel):
    content: str = Field(description="要保存的内容")


@tool("save_to_file", args_schema=FileInput)
def save_to_file(content: str) -> str:
    """把文本内容保存到本地文件 result.txt"""
    print(f"【工具调用】执行文件保存：{content[:20]}...")
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(content)
    return "文件已保存成功"


# 工具列表
tools = [get_weather, calculator, save_to_file]

# ----------------------
# 3. 创建智能体
# ----------------------
agent = create_agent(llm, tools)

# ----------------------
# 4. 流式执行 + 打印完整流程
# ----------------------
if __name__ == "__main__":
    prompt = "先算 100+200，再查北京天气，最后把结果保存到文件"

    print("=" * 60)
    print("【用户提问】", prompt)
    print("=" * 60)

    # 流式执行，能看到每一步
    for step in agent.stream(
            {"messages": [("user", prompt)]},
            stream_mode="values"
    ):
        # 获取最后一条消息
        msg = step["messages"][-1]

        if msg.type == "human":
            print(f"\n🤍 用户：{msg.content}")

        elif msg.type == "ai":
            if not msg.tool_calls:
                # AI 思考/总结
                print(f"\n🧠 LLM思考/总结：{msg.content}")
            else:
                # AI 决定调用工具
                print(f"\n⚙️ LLM决定调用工具：{msg.tool_calls}")

        elif msg.type == "tool":
            # 工具返回结果
            print(f"✅ 工具返回结果：{msg.content}")

    print("\n🎉 智能体流程全部执行完成！")
