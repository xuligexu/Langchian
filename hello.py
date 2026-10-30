import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# 记录api_key,http请求地址
client = OpenAI(
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL")
)

# 调用，确定模型，消息，输出方式
print("正在调用大模型...")
response = client.chat.completions.create(
    model = "qwen3.8-max",
    messages = [
        {
            "role": "system",
            "content": ""   
        },
        {
            "role": "user",
            "content": ""
        }
    ],
    temperature = 0,
    stream = False
)
print("成功调用大模型...")

print(response.choices[0].message.content)