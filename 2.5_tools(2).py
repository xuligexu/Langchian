from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


class Reference(BaseModel):
    title: str = Field(description="The title of the web page cited in the answer")
    url: str = Field(description="The url of the web page cited in the answer.")

class AnswerInfo(BaseModel):
    answer: str = Field(description="The final answer for user.")
    reference: list[Reference] = Field(description="The web pages cited in the answer.")

search_tool = TavilySearch(
    max_results = 2,
    topic = "general",
    tavily_api_key = os.getenv("TAVILY_API_KEY")
)

@tool
def web_search(query: str):
    """Search the web for information."""
    return search_tool.invoke(query)


llm = init_chat_model(
    model = "qwen3.8-max",
    model_provider = "openai",
    api_key = os.getenv("AGENT_API_KEY"),
    base_url = os.getenv("BASE_URL") 
)

agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt="你是一个智能助手，你使用工具来解决用户问题。"
)

response = agent.invoke(
    {
        "messages": [
            HumanMessage("2026年有什么大事，列举2条就行")
        ]
    }
)

for message in response['messages']:
    message.pretty_print()