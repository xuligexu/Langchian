import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEL_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

ChatDeepSeek(
    model = "",
    api_key = "",
)