from typing import Dict, Any, List
from .base_tool import BaseTool
import time
import json
from dataclasses import dataclass
from enum import Enum
from ..utils.llm_client import LLMClientFactory, ModelProvider

class GenerationStrategy(Enum):
    """生成策略枚举"""
    DIRECT = "direct"           # 直接生成
    PROGRESSIVE = "progressive" # 渐进式生成
    ANALYTICAL = "analytical"   # 分析式生成

@dataclass
class GenerationContext:
    """生成上下文"""
    original_query: str
    collected_facts: List[Dict[str, Any]]
    sub_queries: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None

class GenerateAnswerTool(BaseTool):
    """答案生成工具，整合检索到的信息生成最终答案"""
    
    def _setup(self) -> None:
        """初始化配置"""
        self.required_fields = ["context"]
        self.optional_fields = [
            "strategy",      # 生成策略
            "max_length",    # 最大长度
            "temperature"    # 生成温度
        ]
        # 使用统一的LLM客户端，这里使用 deepseek-reasoner 因为需要推理能力
        self.llm_client = LLMClientFactory.get_client(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-reasoner"
        )

    def _get_generation_prompt(self, context: GenerationContext) -> str:
        """生成提示词"""
        facts_summary = "\n".join(
            f"- Source({fact['source']}): {fact['content']}"
            for fact in context.collected_facts
        )
        
        return f"""基于以下信息生成答案。
保持答案的准确性和可靠性，并注明信息来源。

原始查询: {context.original_query}
子查询列表: {context.sub_queries if context.sub_queries else "无"} 
已知信息:
{facts_summary}

回答要求:
1. 确保答案准确且有依据
2. 保持逻辑连贯性
3. 适当组织段落结构
4. 在必要时指出信息的局限性

请以JSON格式回复，包含:
{{
    "answer": "完整的答案",
    "reasoning_chain": ["推理步骤1", "推理步骤2", ...],
    "sources": ["使用的来源1", "使用的来源2", ...],
    "confidence": 0.95  // 置信度评分
}}"""

    def _calculate_confidence(self, facts: List[Dict[str, Any]], sources: List[str]) -> float:
        """计算生成答案的置信度"""
        # 1. 根据事实覆盖率计算
        source_coverage = len(set(sources)) / len(set(f["source"] for f in facts))
        
        # 2. 根据事实一致性计算
        fact_confidences = [
            fact.get("confidence", 0.5) 
            for fact in facts
        ]
        fact_confidence = sum(fact_confidences) / len(fact_confidences)
        
        # 3. 综合评分
        return (source_coverage + fact_confidence) / 2
    
    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行生成"""
        try:
            # 1. 准备生成上下文
            context = GenerationContext(
                original_query=input_data["context"]["original_query"],
                collected_facts=input_data["context"]["collected_facts"],
                sub_queries=input_data["context"].get("sub_queries"),
                metadata=input_data["context"].get("metadata", {})
            )
            
            # 2. 生成提示词
            prompt = self._get_generation_prompt(context)
            
            # 3. 调用LLM生成答案
            response = self.llm_client.chat(
                messages=[{
                    "role": "system",
                    "content": "你是一个专业的答案生成器，善于整合信息并生成准确、连贯的答案。"
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=input_data.get("temperature", 0.7),
                response_format={ "type": "json_object" }
            )
            
            # 4. 解析结果，使用我们统一的响应格式
            result = json.loads(response.content)
            
            # 5. 计算置信度
            confidence = self._calculate_confidence(
                context.collected_facts,
                result["sources"]
            )
            
            # 6. 如果使用了 reasoner 模型，添加推理过程
            if response.metadata.get("has_reasoning"):
                result["detailed_reasoning"] = response.metadata["reasoning"]
            
            return {
                "success": True,
                "data": {
                    "answer": result["answer"],
                    "confidence": confidence,
                    "reasoning_chain": result["reasoning_chain"],
                    "sources": result["sources"],
                    "detailed_reasoning": result.get("detailed_reasoning")
                },
                "error": None,
                "metadata": {
                    "original_query": context.original_query,
                    "generation_strategy": input_data.get("strategy", "direct"),
                    "fact_count": len(context.collected_facts),
                    "model": response.model,
                    "latency": response.latency,
                    "tokens": response.tokens,
                    "timestamp": time.time()
                }
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "data": None,
                "error": f"JSON解析错误: {str(e)}",
                "metadata": {
                    "raw_response": response.content if 'response' in locals() else None,
                    "error_type": "JSONDecodeError"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "metadata": {
                    "error_type": type(e).__name__,
                    "context": str(input_data)
                }
            }

# 使用示例
if __name__ == "__main__":
    # 创建生成工具实例
    generator = GenerateAnswerTool(
        name="generate",
        description="基于检索到的信息生成最终答案"
    )
    
    # 测试上下文
    test_context = {
        "original_query": "What films did Christopher Nolan direct?",
        "collected_facts": [
            {
                "content": "Christopher Nolan directed Inception in 2010",
                "confidence": 0.95,
                "source": "knowledge_graph"
            },
            {
                "content": "Nolan's latest film is Oppenheimer (2023)",
                "confidence": 0.98,
                "source": "vector_search"
            }
        ],
        "sub_queries": [
            {"id": "1", "query": "List Nolan's films"},
            {"id": "2", "query": "What's Nolan's latest film"}
        ]
    }
    
    # 执行生成
    result = generator.execute({
        "context": test_context,
        "strategy": "analytical",
        "temperature": 0.7
    })
    
    print("\n生成结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))