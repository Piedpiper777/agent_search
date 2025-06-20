"""定义了消息的基础结构和相关操作，message是agent与用户、工具之间的通信单元。"""
# 该文件包含消息角色、动作的枚举类型，以及消息类的定义和相关方法。
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import uuid
import time
from enum import Enum
import numpy as np

# 导入我们的自定义类型
from .custom_types import (
    AgentAction, MessageRole, MessageAction, OptimizationMethod,
    ObjectiveVector, ParetoSolution, QuantitativeState, DecisionResult,
    PerformanceMetrics, JSON, MetaData
)

@dataclass
class Message:
    """消息类，表示Agent与用户、工具之间的通信单元"""
    role: MessageRole
    """消息角色"""
    content: Union[str, Dict[str, Any]]
    """消息内容，可以是字符串或字典结构"""
    action: Optional[MessageAction] = None
    """消息动作，可选"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据，存储附加信息"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """会话ID"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """查询ID"""
    timestamp: float = field(default_factory=time.time)
    """时间戳"""
    parent_id: Optional[str] = None
    """父消息ID"""
    
    # ===== 优化相关字段 =====
    optimization_data: Optional[Dict[str, Any]] = field(default_factory=dict)
    """优化算法相关数据"""
    performance_metrics: Optional[PerformanceMetrics] = None
    """性能指标"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "role": self.role.value,
            "action": self.action.value if self.action else None,
            "content": self.content,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
            "optimization_data": self.optimization_data,
            "performance_metrics": self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建消息实例"""
        return cls(
            role=MessageRole(data["role"]),
            action=MessageAction(data["action"]) if data.get("action") else None,
            content=data["content"],
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id", str(uuid.uuid4())),
            query_id=data.get("query_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            parent_id=data.get("parent_id"),
            optimization_data=data.get("optimization_data", {}),
            performance_metrics=data.get("performance_metrics")
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    # ===== 优化数据管理 =====
    def add_optimization_data(self, key: str, value: Any) -> None:
        """添加优化相关数据"""
        if self.optimization_data is None:
            self.optimization_data = {}
        self.optimization_data[key] = value
    
    def get_optimization_data(self, key: str, default: Any = None) -> Any:
        """获取优化相关数据"""
        if self.optimization_data is None:
            return default
        return self.optimization_data.get(key, default)
    
    def set_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """设置性能指标"""
        self.performance_metrics = metrics
    
    def add_pareto_solutions(self, solutions: List[ParetoSolution]) -> None:
        """添加帕累托解集"""
        self.add_optimization_data("pareto_solutions", [
            {
                "action": sol.action.value,
                "objectives": sol.objectives.tolist() if isinstance(sol.objectives, np.ndarray) else sol.objectives,
                "expected_reward": sol.expected_reward,
                "confidence": sol.confidence,
                "policy_path": [a.value for a in sol.policy_path]
            } for sol in solutions
        ])
    
    def add_decision_result(self, decision: DecisionResult) -> None:
        """添加决策结果"""
        self.add_optimization_data("decision_result", {
            "chosen_action": decision["chosen_action"].value,
            "reasoning": decision["reasoning"],
            "confidence": decision["confidence"],
            "method_used": decision["method_used"].value,
            "execution_time": decision["execution_time"]
        })
    
    def add_quantitative_state(self, state: QuantitativeState) -> None:
        """添加量化状态"""
        self.add_optimization_data("quantitative_state", {
            "stage": state.stage,
            "facts_completeness": state.facts_completeness,
            "query_complexity": state.query_complexity,
            "search_exhaustion": state.search_exhaustion,
            "time_elapsed": state.time_elapsed,
            "confidence_level": state.confidence_level
        })

    # ===== 创建子消息方法 =====
    def create_child_message(self, role: MessageRole, action: Optional[MessageAction], 
                           content: Union[str, Dict[str, Any]], 
                           metadata: Optional[Dict[str, Any]] = None) -> 'Message':
        """创建子消息"""
        return Message(
            role=role,
            action=action,
            content=content,
            metadata=metadata or {},
            session_id=self.session_id,
            query_id=str(uuid.uuid4()),
            parent_id=self.query_id
        )

    def create_tool_result_message(
        self, 
        action: MessageAction, 
        result_data: Any,
        execution_time: float,
        quality_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """创建工具执行结果消息"""
        tool_metadata = metadata or {}
        tool_metadata.update({
            "execution_time": execution_time,
            "quality_score": quality_score,
            "success": True
        })
        
        return Message(
            role=MessageRole.TOOL,
            action=action,
            content={
                "result": result_data,
                "type": "tool_execution_result"
            },
            metadata=tool_metadata,
            session_id=self.session_id,
            query_id=str(uuid.uuid4()),
            parent_id=self.query_id
        )

    # ===== 控制器决策消息创建 =====
    def create_controller_decision_message(
        self,
        chosen_action: AgentAction,
        pareto_solutions: List[ParetoSolution],
        decision_result: DecisionResult,
        quantitative_state: QuantitativeState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """创建控制器决策消息"""
        controller_msg = Message(
            role=MessageRole.CONTROLLER,
            action=self._convert_agent_to_message_action(chosen_action),
            content={
                "chosen_action": chosen_action.value,
                "reasoning": decision_result["reasoning"],
                "optimization_method": decision_result["method_used"].value,
                "type": "controller_decision"
            },
            metadata=metadata or {},
            session_id=self.session_id,
            query_id=str(uuid.uuid4()),
            parent_id=self.query_id
        )
        
        # 添加优化数据
        controller_msg.add_pareto_solutions(pareto_solutions)
        controller_msg.add_decision_result(decision_result)
        controller_msg.add_quantitative_state(quantitative_state)
        
        return controller_msg

    def _convert_agent_to_message_action(self, agent_action: AgentAction) -> MessageAction:
        """转换AgentAction到MessageAction"""
        mapping = {
            AgentAction.DECOMPOSE: MessageAction.DECOMPOSE,
            AgentAction.SEARCH_KG: MessageAction.SEARCH_KG,
            AgentAction.SEARCH_VECTOR: MessageAction.SEARCH_VECTOR,
            AgentAction.GENERATE: MessageAction.GENERATE,
            AgentAction.REFINE: MessageAction.REFINE,
            AgentAction.TERMINATE: MessageAction.RESPOND
        }
        return mapping.get(agent_action, MessageAction.RESPOND)

    # ===== 工厂方法 =====
    @classmethod
    def create_user_query(cls, query: str, metadata: Optional[Dict[str, Any]] = None) -> "Message":
        """创建用户查询消息的工厂方法"""
        return cls(
            role=MessageRole.USER,
            action=MessageAction.QUERY,  # 用户查询有明确的action
            content=query,  # 直接使用字符串
            metadata=metadata or {}
        )
    
    @classmethod
    def create_system_message(cls, event: str, data: Dict[str, Any], 
                            metadata: Optional[Dict[str, Any]] = None) -> "Message":
        """创建系统消息"""
        return cls(
            role=MessageRole.SYSTEM,
            action=None,
            content={
                "event": event,
                "data": data,
                "type": "system_event"
            },
            metadata=metadata or {}
        )
    
    @classmethod
    def create_assistant_response(cls, response: str, sources: Optional[List[str]] = None,
                                confidence: float = 0.8, 
                                metadata: Optional[Dict[str, Any]] = None) -> "Message":
        """创建助手回复消息"""
        response_metadata = metadata or {}
        response_metadata.update({
            "confidence": confidence,
            "sources": sources or [],
            "final_answer": True
        })
        
        return cls(
            role=MessageRole.ASSISTANT,
            action=MessageAction.RESPOND,
            content=response,
            metadata=response_metadata
        )

    # ===== 便捷方法 =====
    def is_user_query(self) -> bool:
        """判断是否为用户查询"""
        return self.role == MessageRole.USER and self.action == MessageAction.QUERY
    
    def is_controller_decision(self) -> bool:
        """判断是否为控制器决策"""
        return self.role == MessageRole.CONTROLLER
    
    def is_tool_result(self) -> bool:
        """判断是否为工具结果"""
        return self.role == MessageRole.TOOL
    
    def is_final_answer(self) -> bool:
        """判断是否为最终答案"""
        return (self.role == MessageRole.ASSISTANT and 
                self.action == MessageAction.RESPOND and
                self.get_metadata("final_answer", False))
    
    def get_execution_time(self) -> Optional[float]:
        """获取执行时间"""
        return self.get_metadata("execution_time")
    
    def get_quality_score(self) -> Optional[float]:
        """获取质量分数"""
        return self.get_metadata("quality_score")
    
    def get_optimization_method(self) -> Optional[OptimizationMethod]:
        """获取使用的优化方法"""
        method_str = self.get_optimization_data("decision_result", {}).get("method_used")
        if method_str:
            return OptimizationMethod(method_str)
        return None

    def __str__(self) -> str:
        """字符串表示"""
        action_str = f" [{self.action.value}]" if self.action else ""
        content_preview = str(self.content)[:100] + "..." if len(str(self.content)) > 100 else str(self.content)
        return f"Message({self.role.value}{action_str}): {content_preview}"
    
    def __repr__(self) -> str:
        return self.__str__()


# ===== 消息验证器 =====
class MessageValidator:
    """消息验证器"""
    
    @staticmethod
    def validate_message(message: Message) -> Dict[str, Any]:
        """验证消息完整性"""
        errors = []
        warnings = []
        
        # 基本字段检查
        if not message.role:
            errors.append("消息角色不能为空")
        
        if not message.content:
            errors.append("消息内容不能为空")
        
        # 角色与动作匹配检查
        if message.role == MessageRole.USER and message.action != MessageAction.QUERY:
            warnings.append("用户消息通常应该是QUERY动作")
        
        if message.role == MessageRole.ASSISTANT and message.action != MessageAction.RESPOND:
            warnings.append("助手消息通常应该是RESPOND动作")
        
        # 内容格式检查
        if message.role == MessageRole.CONTROLLER and not isinstance(message.content, dict):
            warnings.append("控制器消息内容建议使用字典格式")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# ===== 使用示例 =====
if __name__ == "__main__":
    # 创建用户查询
    user_msg = Message.create_user_query(
        "What films did Christopher Nolan direct?",
        metadata={"user_id": "user123", "session_start": True}
    )
    print("用户查询:", user_msg)
    
    # 创建控制器决策消息（模拟）
    from .custom_types import QuantitativeState, ParetoSolution, DecisionResult
    
    # 模拟数据
    quant_state = QuantitativeState(
        stage=1, facts_completeness=0.2, query_complexity=0.8,
        search_exhaustion=0.0, time_elapsed=0.1, confidence_level=0.5
    )
    
    pareto_solutions = [
        ParetoSolution(
            action=AgentAction.SEARCH_KG,
            state=quant_state,
            objectives=np.array([0.8, 0.6, 0.7, -0.2]),
            policy_path=[AgentAction.SEARCH_KG, AgentAction.GENERATE],
            expected_reward=0.75,
            confidence=0.85
        )
    ]
    
    decision_result = {
        "chosen_action": AgentAction.SEARCH_KG,
        "reasoning": "基于DP分析，KG搜索能提供最高准确性",
        "confidence": 0.85,
        "method_used": OptimizationMethod.HYBRID,
        "execution_time": 0.05
    }
    
    controller_msg = user_msg.create_controller_decision_message(
        chosen_action=AgentAction.SEARCH_KG,
        pareto_solutions=pareto_solutions,
        decision_result=decision_result,
        quantitative_state=quant_state
    )
    print("\n控制器决策:", controller_msg)
    
    # 创建工具结果消息
    tool_msg = controller_msg.create_tool_result_message(
        action=MessageAction.SEARCH_KG,
        result_data={
            "facts": [
                {"entity": "Christopher Nolan", "relation": "directed", "object": "Inception"},
                {"entity": "Christopher Nolan", "relation": "directed", "object": "Interstellar"}
            ]
        },
        execution_time=2.3,
        quality_score=0.9
    )
    print("\n工具结果:", tool_msg)
    
    # 创建最终回复
    assistant_msg = Message.create_assistant_response(
        "Christopher Nolan has directed many acclaimed films including Inception (2010) and Interstellar (2014)...",
        sources=["knowledge_graph"],
        confidence=0.9
    )
    print("\n助手回复:", assistant_msg)
    
    # 验证消息
    validator = MessageValidator()
    validation_result = validator.validate_message(user_msg)
    print("\n验证结果:", validation_result)