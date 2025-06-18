"""Agent状态管理模块"""
# 该模块定义了Agent的状态类，用于跟踪查询、消息历史、收集的事实等信息
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from message import Message, MessageRole, MessageAction

@dataclass
class AgentState:
    """Agent状态类"""
    history: List[Message] = field(default_factory=list)
    """消息列表，记录了Agent与用户或其他系统的交互历史，class=List[Message]"""
    current_message: Optional[Message] = None
    """当前正在处理的消息，可能是最新的用户输入或系统响应，class=Optional[Message]"""
    collected_facts: List[Dict] = field(default_factory=list)
    """一个字典列表，记录了从不同来源收集的事实信息，class=List[Dict]"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据，存储附加信息，如查询来源、处理状态等，class=Dict[str, Any]"""

    @property
    def origin_query(self) -> Optional[str]:
        """获取原始查询"""
        user_msg = self.get_last_message_by_role(MessageRole.USER)
        return user_msg.content.get("query") if user_msg else None

    @property
    def current_query(self) -> Optional[str]:
        """获取当前正在处理的查询"""
        if self.current_message:
            return self.current_message.content.get("query")
        return self.origin_query

    @property
    def subqueries(self) -> List[str]:
        """获取子查询列表"""
        decompose_msg = self.get_last_message_by_action(MessageAction.DECOMPOSE)
        if decompose_msg:
            return [sq["query"] for sq in decompose_msg.content.get("subqueries", [])]
        return []

    def initialize_with_query(self, query: str) -> None:
        """使用查询初始化状态"""
        self.add_message(Message.create_user_query(query))

    def add_message(self, msg: Message) -> None:
        """添加新消息到历史记录"""
        self.current_message = msg
        self.history.append(msg)
    
    def add_facts(self, facts: List[Dict], source: str) -> None:
        """添加新的事实"""
        for fact in facts:
            fact["source"] = source    #source是fact的一个字段，表示事实的来源
            self.collected_facts.append(fact)
    
    def get_facts_by_source(self, source: str) -> List[Dict]:
        """获取特定来源source的事实"""
        #这里的source指的是事实的来源，比如知识图谱、向量数据库等
        return [f for f in self.collected_facts if f["source"] == source]
    
    def get_last_message_by_role(self, role: MessageRole) -> Optional[Message]:
        """获取最近的特定角色role消息"""
        for msg in reversed(self.history): #时间上最近的，所以是历史消息倒序
            if msg.role == role:
                return msg
        return None
    
    def get_messages_by_action(self, action: MessageAction) -> List[Message]:
        """获取特定动作类型action的所有消息"""
        return [msg for msg in self.history if msg.action == action]
    
    def get_last_message_by_action(self, action: MessageAction) -> Optional[Message]:
        """获取最近的特定动作类型action消息"""
        for msg in reversed(self.history):
            if msg.action == action:
                return msg
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，输入一个AgentState实例，输出一个字典"""
        return {
            "origin_query": self.origin_query,
            "current_query": self.current_query,
            "subqueries": self.subqueries,
            "history": [msg.to_dict() for msg in self.history],
            "current_message": self.current_message.to_dict() if self.current_message else None,
            "collected_facts": self.collected_facts,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """从字典创建状态实例，输入一个字典，输出一个AgentState实例"""
        state = cls(
            metadata=data.get("metadata", {})
        )
        
        # 设置current_query（如果存在）
        if "current_query" in data:
            state.current_query = data["current_query"]
        
        # 恢复消息历史
        for msg_data in data.get("history", []):
            state.add_message(Message.from_dict(msg_data))
            
        # 恢复收集的事实
        state.collected_facts = data.get("collected_facts", [])
        
        # 恢复当前消息
        if data.get("current_message"):
            state.current_message = Message.from_dict(data["current_message"])
            
        return state

    def get_facts_summary(self) -> Dict[str, Any]:
        """获取已收集事实的摘要，包括总事实数量、来源列表和每个来源的事实数量，输入一个AgentState实例，输出一个字典"""
        sources = set(fact["source"] for fact in self.collected_facts)
        return {
            "total_facts": len(self.collected_facts), # 总事实数量
            "sources": list(sources),
            "facts_by_source": {
                source: len(self.get_facts_by_source(source))
                for source in sources
            }
        }

# 使用示例
if __name__ == "__main__":
    # 创建初始状态
    state = AgentState(query="What films did Christopher Nolan direct?")
    
    # 添加一些消息
    decompose_msg = Message(
        role=MessageRole.TOOL,
        action=MessageAction.DECOMPOSE,
        content={
            "subqueries": [
                "What films did Nolan direct in the 2010s?",
                "What are Nolan's most recent films?"
            ]
        }
    )
    state.add_message(decompose_msg)
    
    # 添加一些事实
    kg_facts = [
        {"title": "Inception", "year": 2010},
        {"title": "Interstellar", "year": 2014}
    ]
    state.add_facts(kg_facts, source="knowledge_graph")
    
    # 打印状态摘要
    print("\n状态信息:")
    print(f"查询: {state.query}")
    print(f"消息历史数量: {len(state.history)}")
    print("\n已收集事实摘要:")
    print(state.get_facts_summary())