from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import os

load_dotenv()

# 初始化checkpointer
checkpointer = InMemorySaver()

llm = init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key = os.getenv("AGENT_API_KEY"),
    base_url = os.getenv("BASE_URL")
)

summarize_llm = init_chat_model(
    model = "deepseek-v4-pro-0813",
    model_provider="openai",
    api_key = os.getenv("AGENT_API_KEY"),
    base_url = os.getenv("BASE_URL")
)


# 初始化中间件
middleware = SummarizationMiddleware(
                model=summarize_llm,
                trigger=("messages", 3),
                keep=("messages", 1)
            )

# 创建agent
agent = create_agent(
    model = llm,
    tools=[],
    middleware=[middleware],
    checkpointer=checkpointer
)

config: RunnableConfig = {"configurable": {"thread_id": "thread_3"}}

agent.invoke({"messages": [HumanMessage(content="你好，我是xxx")]}, config)
agent.invoke({"messages": [HumanMessage(content="我最喜欢的运动是排球")]}, config)
agent.invoke({"messages": [HumanMessage(content="我最喜欢的动物是猫")]}, config)
final_response = agent.invoke({"messages": [HumanMessage(content="你还记得我吗")]}, config)

for chunk in final_response["messages"]:
    chunk.pretty_print()