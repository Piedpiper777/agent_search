from typing import Dict, Any, List
from .base_controller import BaseController
from ..core.state import AgentState
from ..utils.llm_client import LLMClientFactory, ModelProvider
import json

class LLMController(BaseController):
    """基于LLM的控制器实现"""
    
    def _setup(self) -> None:
        """初始化LLM控制器"""
        super()._setup()
        # 使用统一的LLM客户端
        self.llm_client = LLMClientFactory.get_client(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-reasoner"  # 使用推理增强的模型
        )
    
    def _get_decision_prompt(self, state: AgentState) -> str:
        """构建决策提示词"""
        # 1. 构建工具描述
        tools_desc = "\n".join(
            f"- {name}: {tool.description}\n  参数: {tool.required_fields}"
            for name, tool in self.tools.items()
        )
        
        # 2. 构建Facts摘要
        facts_summary = "\n".join(
            f"- Source({fact['source']}): {fact['content']}"
            for fact in state.collected_facts
        ) if state.collected_facts else "暂无收集的事实"
        
        # 3. 构建历史摘要
        history_summary = "\n".join(
            f"Step {i+1}: [{msg.role}] -> {msg.action}\n"
            f"  思考: {msg.metadata.get('thought', 'N/A')}\n"
            f"  结果: {msg.content}"
            for i, msg in enumerate(state.history)
        ) if state.history else "暂无执行历史"
        
        return f"""作为智能Agent的决策者，你需要基于当前状态决定下一步操作。

当前查询: {state.query}

可用工具:
{tools_desc}

当前状态:
- 已收集Facts数量: {len(state.collected_facts)}
- 已执行步骤: {len(state.history)}
- 上次动作: {state.current_message.action if state.current_message else 'N/A'}

已收集的Facts:
{facts_summary}

执行历史:
{history_summary}

请分析当前状态并决定下一步动作:
1. 是否需要分解查询？
2. 是否需要收集更多信息？
3. 已有信息是否足够生成答案？
4. 是否需要尝试其他检索策略？

必须以JSON格式回复，包含:
{{
    "thought": "详细的推理过程",
    "action": "要使用的工具名称",
    "tool_name": "同action",
    "input": {{...}},  # 工具所需的输入参数
    "confidence": 0.95  # 决策置信度
}}"""

    def decide_next_action(self, state: AgentState) -> Dict[str, Any]:
        """使用LLM决定下一步操作"""
        try:
            # 1. 生成决策提示词
            prompt = self._get_decision_prompt(state)
            
            # 2. 调用LLM获取决策
            response = self.llm_client.chat(
                messages=[{
                    "role": "system",
                    "content": "你是一个专业的决策分析器，善于基于当前状态规划下一步行动。"
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                response_format={ "type": "json_object" }
            )
            
            # 3. 解析决策结果
            result = json.loads(response.content)
            
            # 4. 验证决策有效性
            self._validate_decision(result)
            
            return {
                "action": result["action"],
                "tool_name": result["tool_name"],
                "input": result["input"],
                "thought": result["thought"],
                "confidence": result["confidence"]
            }
            
        except Exception as e:
            # 决策失败时的后备策略
            return self._fallback_decision(state, str(e))
    
    def _validate_decision(self, decision: Dict[str, Any]) -> None:
        """验证决策的有效性"""
        required_fields = ["action", "tool_name", "input", "thought", "confidence"]
        if not all(field in decision for field in required_fields):
            raise ValueError(f"决策缺少必要字段: {required_fields}")
            
        if decision["tool_name"] not in self.tools:
            raise ValueError(f"未知的工具: {decision['tool_name']}")
            
        tool = self.tools[decision["tool_name"]]
        if not all(field in decision["input"] for field in tool.required_fields):
            raise ValueError(f"工具输入缺少必要字段: {tool.required_fields}")
    
    def _fallback_decision(self, state: AgentState, error: str) -> Dict[str, Any]:
        """决策失败时的后备策略"""
        # 如果还没有执行过分解，则先分解
        if not any(msg.action == "decompose" for msg in state.history):
            return {
                "action": "decompose",
                "tool_name": "decompose",
                "input": {"query": state.query},
                "thought": "决策失败，从最基础的查询分解开始",
                "confidence": 0.5
            }
        
        # 如果已有一些facts但还没生成答案，则生成答案
        if state.collected_facts and not any(msg.action == "generate" for msg in state.history):
            return {
                "action": "generate",
                "tool_name": "generate",
                "input": {
                    "context": {
                        "original_query": state.query,
                        "collected_facts": state.collected_facts
                    }
                },
                "thought": "使用已收集的信息尝试生成答案",
                "confidence": 0.5
            }
        
        # 最后的后备选项：使用向量搜索
        return {
            "action": "search_vector",
            "tool_name": "search_vector",
            "input": {"query": state.query},
            "thought": "使用向量搜索作为最后的尝试",
            "confidence": 0.3
        }

# 使用示例
if __name__ == "__main__":
    from ..tools.decompose import DecomposeTool
    from ..tools.search.search_kg import KGSearchTool
    from ..tools.search.search_vector import VectorSearchTool
    from ..tools.generate import GenerateAnswerTool
    
    # 创建控制器实例
    controller = LLMController()
    
    # 注册工具
    controller.register_tool(DecomposeTool("decompose", "查询分解工具"))
    controller.register_tool(KGSearchTool("search_kg", "知识图谱搜索"))
    controller.register_tool(VectorSearchTool("search_vector", "向量检索"))
    controller.register_tool(GenerateAnswerTool("generate", "答案生成"))
    
    # 执行查询
    final_state = controller.run("What films did Christopher Nolan direct?")
    
    # 打印执行历史
    print("\n执行历史:")
    for msg in final_state.history:
        print(f"\n- Action: {msg.action}")
        print(f"  Thought: {msg.metadata.get('thought', 'N/A')}")
        print(f"  Content: {msg.content}")