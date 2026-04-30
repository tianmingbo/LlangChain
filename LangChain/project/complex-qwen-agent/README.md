# Complex Qwen Agent (LangChain 1.2.x)

一个较复杂的 Agent 示例工程（最新版 `langchain==1.2.12` 风格）：

- 三阶段架构：`Planner -> Executor(with tools) -> Reviewer`
- 工具链：天气、网页搜索、URL 抽取、安全计算、文件读写、时间
- 流式中间过程：`astream_events(version="v2")`
- 会话记忆：Redis 持久化最近任务
- 产出：自动写入 `outputs/report_*.md`

## 目录

- `complex_agent.py` 主程序
- `outputs/` 报告目录

## 依赖

```bash
pip install -r requirements.txt
```

## 环境变量

```bash
cp .env.example .env
# 必填
# DASHSCOPE_API_KEY=...

# Redis 会话历史
# REDIS_URL=redis://localhost:6379/0
# REDIS_KEY_PREFIX=complex-agent:sessions
```

## 运行

```bash
python complex_agent.py \
  --session market-research \
  --goal "分析未来3个月AI编码助手在中小团队落地的可行策略，并给出执行清单"
```

## 说明

- 如果工具调用较多，执行时间会比较长。
- `web_search` 使用 DuckDuckGo Instant API（无需 key，但结果可能不如付费搜索完整）。
- 若要接入你自己的搜索/企业 API，可直接在 `TOOLS` 列表新增工具函数。
- 会话历史保存在 Redis List，按 session_id 分 key，默认最多保留 50 条。
- 天气工具已改为中国大陆公开接口：`http://t.weather.itboy.net/api/weather/city/{city_code}`。
