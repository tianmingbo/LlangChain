create_tool_calling_agent 本质是把「LLM + 提示词 + 工具列表」组装成一个可循环决策的 Runnable Agent，让模型按工具调用协议输出 tool_calls，再由
  AgentExecutor 去真正执行工具并回填结果。

  简化伪码：

  def create_tool_calling_agent(llm, tools, prompt):
      # 1) 把工具声明绑定到模型（函数签名/名称/描述）
      llm_with_tools = llm.bind_tools(tools)

      # 2) 定义单轮“决策链”
      # 输入: user input + scratchpad(历史工具调用轨迹)
      # 输出: 要么是 tool_calls，要么是 final answer
              input=state["input"],
              agent_scratchpad=state["scratchpad"],  # 之前的Action/Observation
          )
          ai_msg = llm_with_tools.invoke(messages)

          if ai_msg.tool_calls:
              # 交给执行器执行工具
              return AgentAction(tool_calls=ai_msg.tool_calls)
          else:
              # 直接结束
              return AgentFinish(output=ai_msg.content)

      return agent_step

  AgentExecutor 运行循环大致是：

  def run_agent(agent, tools, user_input):
      state = {"input": user_input, "scratchpad": []}

      while True:
          decision = agent(state)

          if isinstance(decision, AgentFinish):
          # decision 是多个 tool_calls（可串行/并行执行）
          observations = []
          for call in decision.tool_calls:
              tool = tools[call.name]
              result = tool.invoke(call.args)
              observations.append((call, result))

          # 把本轮工具调用与结果写回 scratchpad，供下一轮推理
          state["scratchpad"].extend(format_as_messages(observations))

  一句话：create_tool_calling_agent 负责“让模型会决定调工具”，AgentExecutor 负责“真正执行工具并循环到结束”。
