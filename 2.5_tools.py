"""
自定义工具
"""


from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


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
def weather(location: str, units: str = "clesius", include_forecast: bool = False):
    """
    Get current weather and optional forecast.
    """
    temp = 22 if units == "celsius" else 72
    result = f"Current Weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days Sunny"
    return result

@tool
def square_root(x: float) -> float:
    """
    Calculate the square root of a number
    """
    return x ** 0.5


llm = init_chat_model(
    model = "qwen3.8-max",
    model_provider = "openai",
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL") 
)

agent = create_agent(
    model=llm,
    tools=[weather, square_root]
)

response = agent.stream(
    {
        "messages": [
            HumanMessage("杭州接下来的天气如何？")
        ]
    },
    stream_mode="messages"
)

for token, metadata in response:
    print(token.content, end = "", flush = True)