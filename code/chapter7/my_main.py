from dotenv import load_dotenv
from my_llm import MyLLM

load_dotenv()

# 实例化自己llm客户端
llm = MyLLM(provider="siliconflow")

# 准备消息
messages = [{"role":"user","content":"你好，请介绍下你自己。"}]

# 发起调用(!这里拿到的是元think方法的迭代器，下面循环就能拿到)
response_stream = llm.think(messages)

# 打印
print("SiliconFlow Response:")
for chunk in response_stream:
    pass
