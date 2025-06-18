from typing import Dict, Any, List, Optional, Union
from enum import Enum
from openai import OpenAI
import os
from abc import ABC, abstractmethod
import json
import time
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """LLM响应结构"""
    content: str
    model: str
    tokens: int
    latency: float # 响应延迟，单位为秒
    raw_response: Any = None # 原始响应内容
    metadata: Dict[str, Any] = None

class ModelProvider(Enum):
    """模型提供商"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # 用于本地部署的模型

class LLMClientBase(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成文本"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """对话形式生成"""
        pass

class OpenAIClient(LLMClientBase):
    """OpenAI API客户端"""
    def __init__(self, model_name: str = "gpt-3.5-turbo"): # 默认使用 GPT-3.5 Turbo 模型
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        latency = time.time() - start_time
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model_name,
            tokens=response.usage.total_tokens,
            latency=latency,
            raw_response=response
        )
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        latency = time.time() - start_time
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model_name,
            tokens=response.usage.total_tokens,
            latency=latency,
            raw_response=response
        )

class DeepseekClient(LLMClientBase):
    """Deepseek API客户端"""
    def __init__(self, model_name: str = "deepseek-chat"): # 默认使用 Deepseek 的聊天模型
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.model_name = model_name
    
    def _process_response(self, response, start_time: float) -> LLMResponse:
        """处理API响应"""
        latency = time.time() - start_time
        
        # 根据模型类型处理响应
        if self.model_name == "deepseek-reasoner":
            content = response.choices[0].message.content
            reasoning = response.choices[0].message.reasoning_content
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                tokens=response.usage.total_tokens,
                latency=latency,
                raw_response=response,
                metadata={
                    "reasoning": reasoning,
                    "has_reasoning": True
                }
            )
        else:
            # 常规模型处理方式
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                tokens=response.usage.total_tokens,
                latency=latency,
                raw_response=response,
                metadata={
                    "has_reasoning": False
                }
            )
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return self._process_response(response, start_time)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return self._process_response(response, start_time)

class LLMClientFactory:
    """LLM客户端工厂"""
    _instances: Dict[str, LLMClientBase] = {}
    
    @classmethod
    def get_client(
        cls,
        provider: Union[ModelProvider, str],
        model_name: Optional[str] = None,
        **kwargs
    ) -> LLMClientBase:
        """获取LLM客户端实例"""
        if isinstance(provider, str):
            provider = ModelProvider(provider)
            
        client_key = f"{provider.value}_{model_name}"
        
        if client_key not in cls._instances:
            if provider == ModelProvider.OPENAI:
                cls._instances[client_key] = OpenAIClient(model_name or "gpt-3.5-turbo")
            elif provider == ModelProvider.DEEPSEEK:
                cls._instances[client_key] = DeepseekClient(model_name or "deepseek-chat")
            # TODO: 添加其他模型提供商的支持
            else:
                raise ValueError(f"不支持的模型提供商: {provider}")
        
        return cls._instances[client_key]

# 使用示例
if __name__ == "__main__":
    # 测试 deepseek-reasoner
    reasoner_client = LLMClientFactory.get_client(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-reasoner"
    )
    
    response = reasoner_client.generate(
        prompt="9.11 and 9.8, which is greater?"
    )
    
    print("\nReasoner结果:")
    print(f"答案: {response.content}")
    if response.metadata.get("has_reasoning"):
        print(f"推理过程: {response.metadata['reasoning']}")
    
    # 测试普通chat模型
    chat_client = LLMClientFactory.get_client(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat"
    )
    
    chat_response = chat_client.generate(
        prompt="What is Python?"
    )
    
    print("\nChat结果:")
    print(chat_response.content)