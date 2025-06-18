from typing import Dict, Any, List
from .base_controller import BaseController
from ..core.state import AgentState
from ..core.message import MessageAction
import time

class RuleController(BaseController):
    """基于规则的控制器实现"""
    
    def _setup(self) -> None:
        """初始化规则控制器"""
        super()._setup()
        # 定义执行流程规则
        self.workflow_rules = {
            "initial": {
                "next": "decompose",
                "thought": "首先需要分解复杂查询"
            },
            "decompose": {
                "next": "search_kg",
                "thought": "使用知识图谱进行精确搜索"
            },
            "search_kg": {
                "condition": lambda facts: len(facts) >= 2,  # 知识图谱找到足够事实
                "success": {
                    "next": "generate",
                    "thought": "已收集足够的事实，可以生成答案"
                },
                "failure": {
                    "next": "search_vector",
                    "thought": "知识图谱结果不足，尝试向量搜索"
                }
            },
            "search_vector": {
                "next": "generate",
                "thought": "使用所有收集的信息生成答案"
            }
        }
    
    def _get_current_stage(self, state: AgentState) -> str:
        """获取当前阶段"""
        if not state.history:
            return "initial"
        return state.current_message.action if state.current_message else "initial"
    
    def _evaluate_condition(self, rule: Dict[str, Any], state: AgentState) -> bool:
        """评估规则条件"""
        if "condition" not in rule:
            return True
            
        if rule["condition"].__code__.co_varnames == ("facts",):
            return rule["condition"](state.collected_facts)
        return rule["condition"](state)
    
    def decide_next_action(self, state: AgentState) -> Dict[str, Any]:
        """基于规则决定下一步操作"""
        try:
            # 1. 获取当前阶段
            current_stage = self._get_current_stage(state)
            
            # 2. 获取规则
            rule = self.workflow_rules.get(current_stage)
            if not rule:
                raise ValueError(f"未知的阶段: {current_stage}")
            
            # 3. 评估条件并获取下一步
            if "condition" in rule:
                success = self._evaluate_condition(rule, state)
                next_rule = rule["success"] if success else rule["failure"]
            else:
                next_rule = rule
            
            # 4. 准备工具输入
            tool_input = self._prepare_tool_input(next_rule["next"], state)
            
            return {
                "action": next_rule["next"],
                "tool_name": next_rule["next"],
                "input": tool_input,
                "thought": next_rule["thought"],
                "confidence": 1.0  # 规则基决策的置信度总是1.0
            }
            
        except Exception as e:
            # 发生错误时的后备决策
            return self._fallback_decision(state, str(e))
    
    def _prepare_tool_input(self, action: str, state: AgentState) -> Dict[str, Any]:
        """准备工具输入"""
        if action == "decompose":
            return {"query": state.query}
            
        elif action == "search_kg":
            # 如果有分解结果，使用第一个子查询
            decompose_msg = next(
                (msg for msg in state.history if msg.action == "decompose"),
                None
            )
            if decompose_msg and "subqueries" in decompose_msg.content:
                return {"query": decompose_msg.content["subqueries"][0]["query"]}
            return {"query": state.query}
            
        elif action == "search_vector":
            # 使用原始查询进行向量搜索
            return {"query": state.query}
            
        elif action == "generate":
            return {
                "context": {
                    "original_query": state.query,
                    "collected_facts": state.collected_facts,
                    "sub_queries": state.get_messages_by_action(MessageAction.DECOMPOSE)
                }
            }
        
        raise ValueError(f"未知的动作类型: {action}")
    
    def _fallback_decision(self, state: AgentState, error: str) -> Dict[str, Any]:
        """决策失败时的后备策略"""
        if not state.history:
            # 如果还没有任何历史，从分解开始
            return {
                "action": "decompose",
                "tool_name": "decompose",
                "input": {"query": state.query},
                "thought": "从查询分解开始",
                "confidence": 0.5
            }
        
        if state.collected_facts:
            # 如果已经收集了一些事实，尝试生成答案
            return {
                "action": "generate",
                "tool_name": "generate",
                "input": {
                    "context": {
                        "original_query": state.query,
                        "collected_facts": state.collected_facts
                    }
                },
                "thought": "使用已收集的信息生成答案",
                "confidence": 0.5
            }
        
        # 最后的尝试：直接使用向量搜索
        return {
            "action": "search_vector",
            "tool_name": "search_vector",
            "input": {"query": state.query},
            "thought": "使用向量搜索作为最后尝试",
            "confidence": 0.3
        }

# 使用示例
if __name__ == "__main__":
    from ..tools.decompose import DecomposeTool
    from ..tools.search.kg import KGSearchTool
    from ..tools.search.vector import VectorSearchTool
    from ..tools.generate import GenerateAnswerTool
    
    # 创建控制器实例
    controller = RuleController()
    
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

# 但是基于规则无法体现出agent的智能决策能力，无法处理复杂的查询和动态变化的环境。
# 因此，基于规则的控制器通常适用于简单、结构化的任务，而对于复杂任务，还是需要更智能的决策机制。