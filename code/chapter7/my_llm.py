import os
from typing import Optional,Iterator
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对SiliconFlow的支持。
    """

    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        # 自定义LLM供应商(非继承父类里HelloAgentsLLM已定义有的)
        if provider == "siliconflow":
            print("正在使用自定义的 SiliconFlow Provider")
            self.provider = "siliconflow"

            # 解析 SiliconFlow 凭证
            self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
            self.base_url = base_url or os.getenv("SILICONFLOW_LLM_BASE_URL")

            # 验证凭证是否存在
            if not self.api_key or not self.base_url:
                raise ValueError(
                    "SiliconFlow API key or BaseUrl not found. Please set SiliconFlow_API_KEY or BaseUrl environment variable."
                )

            # 设置默认模型和其他参数
            self.model = (model or os.getenv("SILICONFLOW_LLM_MODEL_ID") or "deepseek-ai/DeepSeek-V3.2")
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
                )

        else:
            super().__init__(model=model,api_key=api_key,base_url=base_url,provider=provider,**kwargs)

    def think(
        self, messages: list[dict[str, str]], temperature: Optional[float] = None
    ) -> Iterator[str]:
        """
        !!! 自己重写父类think方法，兼容SiliconFlow流式响应中choices可能为空的情况
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """

        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=(
                    temperature if temperature is not None else self.temperature
                ),
                max_tokens=self.max_tokens,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            for chunk in response:
                if not chunk.choices:  # 跳过choices为空的chunk
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise Exception(f"LLM调用失败: {str(e)}")
