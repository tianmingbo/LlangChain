"""Complex agent demo for LangChain 1.2.x + Tongyi (Qwen).

Features:
- Three-stage pipeline: planner -> tool-using executor -> reviewer
- Real tools: weather, web search, URL extract, calculator, file ops, time
- Streamed intermediate events via astream_events
- Session memory persisted to Redis
- Final markdown report saved to outputs/
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import datetime as dt
import glob
import json
import os
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import redis
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class RedisMemoryStore:
    def __init__(self, redis_url: str, key_prefix: str = "complex-agent:sessions"):
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"

    def append(self, session_id: str, payload: dict[str, Any], max_items: int = 50) -> None:
        key = self._key(session_id)
        self.client.lpush(key, json.dumps(payload, ensure_ascii=False))
        self.client.ltrim(key, 0, max_items - 1)

    def load_recent(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        key = self._key(session_id)
        rows = self.client.lrange(key, 0, max(limit - 1, 0))
        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                items.append(json.loads(row))
            except json.JSONDecodeError:
                continue
        return list(reversed(items))


# ----------------------------
# Tool definitions
# ----------------------------


@tool
def get_weather(city: str) -> str:
    """Get real-time weather by city name using Open-Meteo (no API key required)."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=20,
    ).json()
    if not geo.get("results"):
        return f"City not found: {city}"

    place = geo["results"][0]
    data = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto",
        },
        timeout=20,
    ).json()

    cur = data.get("current", {})
    return (
        f"{place.get('name')} ({place.get('country', '')}): "
        f"temp={cur.get('temperature_2m')}°C, "
        f"humidity={cur.get('relative_humidity_2m')}%, "
        f"wind={cur.get('wind_speed_10m')}km/h, "
        f"weather_code={cur.get('weather_code')}"
    )


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo instant answer API. Returns compact snippets."""
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=20,
    )
    data = resp.json()

    out: list[str] = []
    if data.get("AbstractText"):
        out.append(f"Abstract: {data['AbstractText']}")
    if data.get("Answer"):
        out.append(f"Answer: {data['Answer']}")

    related = data.get("RelatedTopics", [])
    count = 0
    for item in related:
        if isinstance(item, dict) and item.get("Text"):
            out.append(f"- {item['Text']}")
            count += 1
        if count >= max_results:
            break

    if not out:
        return "No concise result from DuckDuckGo instant API."
    return "\n".join(out)


@tool
def fetch_url_text(url: str, max_chars: int = 6000) -> str:
    """Fetch URL and return cleaned visible text."""
    resp = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return text[:max_chars]


_ALLOWED_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        return _ALLOWED_BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Safely calculate arithmetic expression, e.g. '(3+5)*2/7'."""
    parsed = ast.parse(expression, mode="eval")
    value = _safe_eval(parsed.body)
    return str(value)


@tool
def list_files(pattern: str = "**/*") -> str:
    """List files under current project using glob pattern."""
    files = [p for p in glob.glob(pattern, recursive=True) if Path(p).is_file()]
    files = files[:200]
    return "\n".join(files) if files else "No files found"


@tool
def read_text_file(path: str, max_chars: int = 8000) -> str:
    """Read text file content with max_chars cap."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return f"File not found: {path}"
    text = p.read_text(encoding="utf-8", errors="ignore")
    return text[:max_chars]


@tool
def write_text_file(path: str, content: str) -> str:
    """Write text content to a file path (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written: {p} ({len(content)} chars)"


@tool
def now_utc() -> str:
    """Get current UTC time in ISO format."""
    return _utc_now()


TOOLS = [
    get_weather,
    web_search,
    fetch_url_text,
    calculate,
    list_files,
    read_text_file,
    write_text_file,
    now_utc,
]


@dataclass
class StepResult:
    step_index: int
    title: str
    output: str


class ComplexAgentApp:
    def __init__(self, model_name: str = "qwen-plus"):
        load_dotenv()
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_key_prefix = os.getenv("REDIS_KEY_PREFIX", "complex-agent:sessions")
        self.memory = RedisMemoryStore(redis_url=redis_url, key_prefix=redis_key_prefix)
        self.model = ChatTongyi(model=model_name, temperature=0)

        self.planner = create_agent(
            model=self.model,
            tools=[],
            system_prompt=textwrap.dedent(
                """
                You are a planning specialist.
                Return ONLY JSON object:
                {
                  "goal": "...",
                  "steps": [{"title": "...", "instruction": "..."}, ...],
                  "risks": ["...", "..."]
                }
                Rules:
                - 4 to 8 steps.
                - instructions must be executable by a tool-using researcher agent.
                - Keep each instruction concise and concrete.
                """
            ).strip(),
        )

        self.executor = create_agent(
            model=self.model,
            tools=TOOLS,
            system_prompt=textwrap.dedent(
                """
                You are an execution researcher.
                - Always use tools when real data is needed.
                - Be explicit about assumptions and sources.
                - Return concise markdown output for each step.
                """
            ).strip(),
        )

        self.reviewer = create_agent(
            model=self.model,
            tools=[],
            system_prompt=textwrap.dedent(
                """
                You are a review-and-synthesis expert.
                Produce final markdown report with sections:
                1) Executive Summary
                2) Findings
                3) Evidence & Tool Trace Highlights
                4) Risks and Unknowns
                5) Next Actions
                Keep it precise and actionable.
                """
            ).strip(),
        )

    async def _run_step_with_events(self, step_idx: int, step_title: str, instruction: str, context: str) -> str:
        step_input = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Step {step_idx}: {step_title}\n"
                        f"Instruction: {instruction}\n\n"
                        f"Current Context:\n{context}\n\n"
                        "Return a markdown section with key findings."
                    ),
                }
            ]
        }

        print(f"\n===== STEP {step_idx}: {step_title} =====")
        async for event in self.executor.astream_events(step_input, version="v2"):
            et = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})
            if et == "on_tool_start":
                print(f"[tool:start] {name} input={data.get('input')}")
            elif et == "on_tool_end":
                out = data.get("output")
                out_s = str(out)
                print(f"[tool:end]   {name} output={out_s[:220]}{'...' if len(out_s) > 220 else ''}")

        result = await self.executor.ainvoke(step_input)
        msg = result["messages"][-1]
        return msg.content if hasattr(msg, "content") else str(msg)

    async def run(self, session_id: str, user_goal: str) -> dict[str, Any]:
        history = self.memory.load_recent(session_id=session_id, limit=5)
        history_text = "\n".join(
            [f"- {h.get('goal', '')} | report={h.get('report_path', 'n/a')}" for h in history]
        ) or "(none)"

        plan_input = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"User goal:\n{user_goal}\n\n"
                        f"Recent session history:\n{history_text}\n"
                    ),
                }
            ]
        }
        plan_raw = await self.planner.ainvoke(plan_input)
        plan_content = plan_raw["messages"][-1].content

        # robust json extraction
        plan_json: dict[str, Any]
        try:
            plan_json = json.loads(plan_content)
        except json.JSONDecodeError:
            start = plan_content.find("{")
            end = plan_content.rfind("}")
            plan_json = json.loads(plan_content[start: end + 1])

        steps = plan_json.get("steps", [])
        if not steps:
            raise RuntimeError("Planner returned no executable steps")

        step_results: list[StepResult] = []
        rolling_context = f"Goal: {plan_json.get('goal', user_goal)}"

        for idx, step in enumerate(steps, start=1):
            output = await self._run_step_with_events(
                step_idx=idx,
                step_title=step["title"],
                instruction=step["instruction"],
                context=rolling_context,
            )
            step_results.append(StepResult(step_index=idx, title=step["title"], output=output))
            rolling_context += f"\n\n[Step {idx} - {step['title']}]\n{output[:2500]}"

        synthesis_payload = {
            "goal": plan_json.get("goal", user_goal),
            "risks": plan_json.get("risks", []),
            "steps": [
                {"index": s.step_index, "title": s.title, "output": s.output}
                for s in step_results
            ],
        }

        review_input = {
            "messages": [
                {
                    "role": "user",
                    "content": "Synthesize final report from this JSON:\n" + json.dumps(
                        synthesis_payload, ensure_ascii=False
                    ),
                }
            ]
        }
        review_raw = await self.reviewer.ainvoke(review_input)
        report_md = review_raw["messages"][-1].content

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = OUTPUT_DIR / f"report_{session_id}_{ts}.md"
        report_path.write_text(report_md, encoding="utf-8")

        result_obj = {
            "session_id": session_id,
            "goal": user_goal,
            "plan": plan_json,
            "step_count": len(step_results),
            "report_path": str(report_path),
            "created_at": _utc_now(),
        }
        self.memory.append(session_id=session_id, payload=result_obj)
        return result_obj


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="Complex Qwen agent demo (LangChain 1.2.x)")
    parser.add_argument("--goal", required=True, help="Research/analysis goal")
    parser.add_argument("--session", default="default", help="Session id for memory")
    parser.add_argument("--model", default="qwen-plus", help="Tongyi model name")
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not found in environment.")
        return 2

    app = ComplexAgentApp(model_name=args.model)

    try:
        result = await app.run(session_id=args.session, user_goal=args.goal)
        print("\n===== DONE =====")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Report saved: {result['report_path']}")
        return 0
    except Exception:
        print("Execution failed:")
        print(traceback.format_exc())
        return 1


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
