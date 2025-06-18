from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from ..core.message import Message, MessageRole, MessageAction
from ..core.state import AgentState
from ..tools.base_tool import BaseTool
import time
import json

class BaseController(ABC):
    """控制器基类，定义了Agent的核心控制逻辑"""
    
    def __init__(self):
        self.state: Optional[AgentState] = None
        self.tools: Dict[str, BaseTool] = {}
        self.max_steps: int = 10  # 最大执行步数
        self._setup()
    
    def _setup(self) -> None:
        """初始化控制器配置，子类可重写此方法"""
        pass
    
    def register_tool(self, tool: BaseTool) -> None:
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self.tools.get(tool_name)
    
    @abstractmethod
    def decide_next_action(self, state: AgentState) -> Dict[str, Any]:
        """决定下一步操作
        
        Returns:
            Dict[str, Any]: {
                "action": str,          # 动作名称
                "tool_name": str,       # 工具名称
                "input": Dict[str, Any], # 工具输入
                "thought": str,         # 推理过程
                "confidence": float     # 决策置信度
            }
        """
        pass
    
    def execute_action(self, action: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """执行决策的动作"""
        tool = self.get_tool(action["tool_name"])
        if not tool:
            raise ValueError(f"未知的工具: {action['tool_name']}")
    
    # 创建工具调用消息
        tool_message = Message(
            role=MessageRole.CONTROLLER,
            action=action["action"],
            content=state.current_query,
            metadata={
                "thought": action["thought"],
                "confidence": action["confidence"]
            }
        )
    
        # 执行工具并获取响应消息
        response_message = tool.execute(tool_message)
    
        return {
            "success": response_message.role != MessageRole.ERROR,
            "message": response_message
        }
    
    def handle_tool_response(
        self, 
        response: Dict[str, Any], 
        state: AgentState
    ) -> AgentState:
        """处理工具返回结果"""
        # 创建新消息
        msg = Message(
            role=MessageRole.TOOL,
            action=response["metadata"]["action"],
            content=response["output"],
            metadata=response["metadata"]
        )
        
        # 更新状态
        state.add_message(msg)
        
        # 如果是搜索结果，添加到收集的事实中
        if "facts" in response["output"]:
            state.add_facts(
                response["output"]["facts"],
                source=response["metadata"]["action"]
            )
        
        return state
    
    def should_continue(self, state: AgentState) -> bool:
        """判断是否应该继续执行"""
        # 检查是否达到最大步数
        if len(state.history) >= self.max_steps:
            return False
        
        # 检查是否已经生成了最终答案
        last_message = state.current_message
        if last_message and last_message.action == MessageAction.GENERATE:
            return False
        
        # 检查是否有严重错误
        if state.metadata.get("critical_error"):
            return False
            
        return True
    
    def run(self, query: str) -> AgentState:
        """运行控制器"""
        # 初始化状态
        self.state = AgentState(query=query)
        
        try:
            while self.should_continue(self.state):
                # 1. 决定下一步操作
                action = self.decide_next_action(self.state)
                
                # 2. 执行操作
                response = self.execute_action(action, self.state)
                
                # 3. 处理响应
                if response["success"]:
                    self.state = self.handle_tool_response(response, self.state)
                else:
                    # 处理错误
                    self.state.metadata["critical_error"] = response["error"]
                    break
                    
            return self.state
            
        except Exception as e:
            # 处理意外错误
            self.state.metadata["critical_error"] = str(e)
            return self.state

# 使用示例
if __name__ == "__main__":
    # 这里创建一个简单的测试控制器
    class TestController(BaseController):
        def decide_next_action(self, state: AgentState) -> Dict[str, Any]:
            # 简单的测试逻辑
            return {
                "action": "decompose",
                "tool_name": "decompose",
                "input": {"query": state.query},
                "thought": "需要先分解查询",
                "confidence": 0.9
            }
    
    from ..tools.decompose import DecomposeTool
    
    # 创建控制器实例
    controller = TestController()
    
    # 注册工具
    decompose_tool = DecomposeTool(
        name="decompose",
        description="查询分解工具"
    )
    controller.register_tool(decompose_tool)
    
    # 执行查询
    final_state = controller.run("What films did Christopher Nolan direct?")
    
    # 打印结果
    print("\n执行历史:")
    for msg in final_state.history:
        print(f"\n- Action: {msg.action}")
        print(f"  Thought: {msg.metadata.get('thought', 'N/A')}")
        print(f"  Content: {msg.content}")