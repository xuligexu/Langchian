from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal
import os


# 这里只为记录tool操作，先忽视参数不匹配问题
class WeatherInput(BaseModel):
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def square_root(x: float) -> float:
    """
    Calculate the square root of a number
    """
    return x ** 0.5

llm = init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL"),
    tools = [square_root]
)
