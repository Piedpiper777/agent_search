"""定义查询分解工具，使用DeepSeek的LLM将复杂查询分解为多个简单子查询"""
from typing import Dict, Any, List
import time
import json
from ..core.custom_types import ToolResult
from .base_tool import BaseTool
from ..utils.llm_client import LLMClientFactory, ModelProvider
from ..config.custom import get_config
from ..core.message import Message, MessageRole, MessageAction

class DecomposeTool(BaseTool):
    """查询分解工具，使用LLM将复杂查询分解为多个简单子查询"""
    
    def _setup(self) -> None:
        """初始化配置"""
        # 从配置系统获取工具参数
        config = get_config()
        self.max_subqueries = config["tools"]["decompose"].get("max_subqueries", 3)
        
        # 只保留必需的消息内容字段
        self.required_fields = ["content"]  # Message中的content字段
        self.optional_fields = []  # 不需要额外的可选字段
        
        # 使用统一的LLM客户端
        self.llm_client = LLMClientFactory.get_client(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-chat"
        )
    
    def _get_decompose_prompt(self, query: str) -> str:
        """生成分解提示词"""
        return f"""请将以下复杂查询分解为多个简单的子查询。
每个子查询应该容易被搜索引擎或知识图谱理解。
同时提供分解的思路解释。

原始查询: {query}

请以JSON格式回复，包含:
{{
    "subqueries": [
        {{
            "id": "子查询ID",
            "query": "子查询内容",
            "reasoning": "为什么需要这个子查询"
        }}
    ],
    "strategy": "总体分解策略说明"
}}"""
# 以prompt的形式规定输出格式

    def _execute(self, message: Message) -> Message:
        """执行查询分解"""
        try:
            # 从消息内容中获取查询
            query = message.content["query"]
            
            # 使用统一客户端进行调用
            response = self.llm_client.chat(
                messages=[{
                    "role": "system",
                    "content": "你是一个专业的问题分解助手，擅长将复杂问题分解为简单的子问题。"
                }, {
                    "role": "user",
                    "content": self._get_decompose_prompt(query)
                }],
                temperature=0.7,
                response_format={ "type": "json_object" }
            )
            
            # 解析响应
            result = json.loads(response.content)
            
            subqueries = result["subqueries"][:self.max_subqueries]  # 限制子查询数量
            
            # 创建响应消息
            return Message(
                role=MessageRole.TOOL,
                action=MessageAction.DECOMPOSE,
                content={
                    "query": query,  # 保持原始查询
                    "subqueries": [
                        {
                            "id": f"sq_{i}",
                            "query": subq["query"],
                            "reasoning": subq["reasoning"]
                        }
                        for i, subq in enumerate(subqueries)
                    ]
                },
                metadata={
                    "model": response.model,
                    "latency": response.latency,
                    "tokens": response.tokens,
                    "thought": "将复杂查询分解为简单子查询"
                },
                parent_id=message.query_id  # 关联到输入消息
            )
            
        except Exception as e:
            # 创建错误消息
            return Message(
                role=MessageRole.ERROR,
                action=MessageAction.DECOMPOSE,
                content=str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "original_query": query
                },
                parent_id=message.query_id
            )

# 使用示例
if __name__ == "__main__":
    # 创建工具实例
    decompose_tool = DecomposeTool(
        name="decompose",
        description="将复杂问题分解为多个简单子问题"
    )
    
    # 测试查询
    test_query = "What films did Christopher Nolan direct and what are their box office earnings and critical reviews?"
    
    result = decompose_tool.execute({
        "query": test_query,
        "max_subqueries": 3
    })
    
    print("\n问题分解结果:")
    if result["success"]:
        print("\n子问题列表:")
        for subquery in result["data"]["subqueries"]:
            print(f"\n- ID: {subquery['id']}")
            print(f"  问题: {subquery['query']}")
            print(f"  原因: {subquery['reasoning']}")
        print(f"\n分解策略: {result['data']['strategy']}")
    else:
        print(f"错误: {result['error']}")