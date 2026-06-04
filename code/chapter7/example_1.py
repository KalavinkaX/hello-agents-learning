from hello_agents import SimpleAgent,HelloAgentsLLM
from dotenv import load_dotenv

load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM(provider="deepseek")

agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手"
)

# basic conversation
response = agent.run("你好！请介绍一下自己")
print(response)

# 添加工具功能(可选)
from hello_agents.tools import CalculatorTool
calculator = CalculatorTool()
# 需要实现7.4.1的MySimpleAgent进行调用，后续章节会支持此类调用方式
# agent.add_tool(calculator)

response = agent.run("请帮我计算 2 + 3 * 4")
print(response)

# 查看对话历史
print(f"历史消息数：{len(agent.get_history())}")
