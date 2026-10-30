from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
import os

load_dotenv()

system_prompt = """
# 身份
- 你是一个科幻作家，根据用户的要求创建一个太空之都。

# 指令
-请务必以 JSON 格式输出，不要加任何markdown样式。

# 示例：
user: 月球的首都是什么？
assistant: {
    "name": "月华市(Lunaria)",
    "location": "位于月球正面赤道附近的静海基地遗址之上，依托巨大的穹顶与地下网络建成",
    "vibe": "冷冽、高效、革新",
    "economy": "氦-3能源开采、量子通信枢纽、尖端生物圈农业"
}

"""

llm = init_chat_model(
    model = "qwen3.8-max",
    model_provider="openai",
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL")
)


agent = create_agent(
    model = llm,
    system_prompt = system_prompt
)

response = agent.invoke(
    {
        "messages": [
            HumanMessage("金星的首都是什么？")
        ]
    }
)

for chunk in response["messages"]:
    chunk.pretty_print()

# print(response['messages'][-1].content)


# --------------------------------------------
# 这种方式不需要额外去填写示例，直接让agent结构化输出

class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

llm = init_chat_model(
    model = "qwen3.8-max",
    model_provider="openai",
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL")
)


agent = create_agent(
    model = llm,
    system_prompt = "你是一个科幻作家，根据用户的要求创建一个太空之都。",
    response_format=CapitalInfo # 设置结构化输出的格式
)