from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

@tool
def getWeather(location: str) -> str:
    """
    Get the weather in a given location.
    Args:
        location: city name or coordinates
    """
    return f"Current weather in {location} is sunny"

# 确定模型
llm = ChatOpenAI(
    model = "qwen3.8-max",
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL"),
    temperature = 0
)

# 创建智能体
agent = create_agent(
    model = llm,
    tools = [getWeather]
)

# 发起调用 
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "杭州今天天气如何？"
            }
        ]
    }
)

# print(response)
for message in response['messages']:
    print(message.model_dump_json(indent=2))
    print("---------------------------------")