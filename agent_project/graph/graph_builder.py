from typing import Dict, Any, List, Callable
from langgraph.graph import StateGraph, END
from ..core.state import AgentState
from ..core.message import Message, MessageRole
from ..controller.llm import LLMController
from ..tools.decompose import DecomposeTool
from ..tools.search.search_kg import KGSearchTool
from ..tools.search.search_vector import VectorSearchTool
from ..tools.generate import GenerateAnswerTool
import json

def create_agent_graph() -> StateGraph:
    """创建Agent的工作流图"""
    
    # 1. 初始化控制器和工具
    controller = LLMController()
    controller.register_tool(DecomposeTool("decompose", "查询分解工具"))
    controller.register_tool(KGSearchTool("search_kg", "知识图谱搜索"))
    controller.register_tool(VectorSearchTool("search_vector", "向量检索"))
    controller.register_tool(GenerateAnswerTool("generate", "答案生成"))

    # 2. 创建工作流图
    workflow = StateGraph(AgentState)

    # 3. 定义节点处理函数
    def decide_next(state: AgentState) -> Dict[str, Any]:
        """决策节点"""
        if not controller.should_continue(state):
            return {"next": END}
            
        action = controller.decide_next_action(state)
        return {"next": action["action"]}

    def execute_tool(state: AgentState, tool_name: str) -> AgentState:
        """工具执行节点"""
        tool = controller.get_tool(tool_name)
        action = controller.decide_next_action(state)
        
        # 只有当决策的工具与当前节点匹配时才执行
        if action["tool_name"] == tool_name:
            response = controller.execute_action(action, state)
            if response["success"]:
                state = controller.handle_tool_response(response, state)
            else:
                state.metadata["error"] = response["error"]
                
        return state

    # 4. 添加决策节点
    workflow.add_node("decide", decide_next)

    # 5. 添加工具节点
    for tool_name in ["decompose", "search_kg", "search_vector", "generate"]:
        workflow.add_node(
            tool_name,
            lambda state, tool=tool_name: execute_tool(state, tool)
        )

    # 6. 设置起始节点
    workflow.set_entry_point("decide")

    # 7. 添加边（转换规则）
    # 从决策节点到各工具节点
    workflow.add_edge("decide", "decompose")
    workflow.add_edge("decide", "search_kg")
    workflow.add_edge("decide", "search_vector")
    workflow.add_edge("decide", "generate")
    
    # 从工具节点回到决策节点
    workflow.add_edge("decompose", "decide")
    workflow.add_edge("search_kg", "decide")
    workflow.add_edge("search_vector", "decide")
    workflow.add_edge("generate", "decide")

    # 8. 编译工作流
    workflow.compile()
    
    return workflow

# 使用示例
if __name__ == "__main__":
    # 创建工作流
    graph = create_agent_graph()
    
    # 创建初始状态
    initial_state = AgentState(
        query="What films did Christopher Nolan direct?"
    )
    
    # 执行工作流
    for state in graph.run(initial_state):
        if isinstance(state, AgentState):
            print("\n当前状态:")
            print(f"- 步骤: {len(state.history)}")
            if state.current_message:
                print(f"- 动作: {state.current_message.action}")
                print(f"- 思考: {state.current_message.metadata.get('thought', 'N/A')}")
    
    # 打印最终结果
    print("\n执行历史:")
    for msg in state.history:
        print(f"\n[{msg.action}]")
        print(f"思考: {msg.metadata.get('thought', 'N/A')}")
        if isinstance(msg.content, dict):
            print(f"结果: {json.dumps(msg.content, indent=2, ensure_ascii=False)}")
        else:
            print(f"结果: {msg.content}")