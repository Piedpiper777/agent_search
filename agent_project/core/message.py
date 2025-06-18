"""定义了消息的基础结构和相关操作，message是agent与用户、工具之间的通信单元。"""
# 该文件包含消息角色、动作的枚举类型，以及消息类的定义和相关方法。
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid
import time
from enum import Enum, auto #导入枚举类和自动赋值功能

class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    """系统配置消息"""
    USER = "user"
    """用户输入消息"""
    CONTROLLER = "controller"
    """控制器决策消息"""
    TOOL = "tool"
    """具执行结果消息"""
    ERROR = "error" 
    """错误处理消息"""
    
class MessageAction(Enum):
    """消息动作枚举"""
    DECOMPOSE = "decompose"
    """分解查询消息，表示将一个查询分解为多个子查询"""
    SEARCH_KG = "search_kg"
    """知识图谱查询消息，表示在知识图谱中搜索相关信息"""
    SEARCH_VECTOR = "search_vector"
    """向量数据库查询消息，表示在向量数据库中搜索相关信息"""
    GENERATE = "generate"
    """生成消息，表示生成新的内容或响应"""
    ERROR_HANDLE = "error_handle"
    """错误处理消息，表示处理错误或异常情况"""
    #后续如添加功能此处需更新
    
    
@dataclass
class Message:
    """消息类，表示Agent与用户、工具之间的通信单元"""
    role: MessageRole
    """消息角色，class=MessageRole"""
    action: MessageAction
    """消息动作，class=MessageAction"""
    content: Dict[str, Any]  # 改为Dict类型,规范内容格式
    """消息内容，class=Dict[str, Any]"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据，存储附加信息，如置信度、推理过程等，class=Dict[str, Any]"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """会话ID，用于标识消息所属的会话，默认为UUID，class=str"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """查询ID，用于标识消息的查询，默认为UUID，class=str"""
    timestamp: float = field(default_factory=time.time)
    """时间戳，表示消息发送的时间，默认为当前时间，class=float"""
    parent_id: Optional[str] = None
    """父消息ID，用于表示消息的上下文关系，默认为None，class=Optional[str]"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "role": self.role.value,
            "action": self.action.value,
            "content": self.content,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建消息实例"""
        return cls(
            role=MessageRole(data["role"]),
            action=MessageAction(data["action"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id", str(uuid.uuid4())),
            query_id=data.get("query_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            parent_id=data.get("parent_id")
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    def create_child_message(self, role: MessageRole, action: MessageAction, 
                           content: Any, metadata: Dict[str, Any] = None) -> 'Message':
        """创建子消息"""
        return Message(
            role=role,
            action=action,
            content=content,
            metadata=metadata or {},
            session_id=self.session_id,  # 继承同一会话ID
            query_id=str(uuid.uuid4()),  # 新的查询ID
            parent_id=self.query_id      # 设置父消息的查询ID为parent_id
        )

    def create_subquery_message(
        self, 
        action: MessageAction, 
        content: Any, 
        metadata: Dict[str, Any] = None
    ) -> 'Message':
        """创建子查询消息"""
        return Message(
            role=MessageRole.TOOL,  # 子查询消息固定为TOOL角色
            action=action,
            content=content,
            metadata=metadata or {},
            session_id=self.session_id,  # 继承同一会话ID
            query_id=str(uuid.uuid4()),  # 新的查询ID
            parent_id=self.query_id      # 设置父消息的查询ID为parent_id
        )

    @classmethod
    def create_user_query(cls, query: str) -> "Message":
        """创建用户查询消息的工厂方法"""
        return cls(
            role=MessageRole.USER,
            action=None,  # 初始查询没有具体动作
            content={
                "query": query,  # 统一使用content["query"]存储查询内容
                "type": "user_query"
            }
        )

# 使用示例
if __name__ == "__main__":
    # 创建一个查询分解消息
    decompose_msg = Message(
        role=MessageRole.TOOL,
        action=MessageAction.DECOMPOSE,
        content={
            "subqueries": [
                "What films did Nolan direct in the 2010s?",
                "What are Nolan's most recent films?"
            ],
            "strategy": "Splitting by time periods"
        },
        metadata={
            "confidence": 0.95,
            "reasoning": "将时间范围分开查询可以获得更精确的结果"
        }
    )
    
    # 转换为字典
    msg_dict = decompose_msg.to_dict()
    print("\n消息字典格式:")
    print(msg_dict)
    
    # 从字典重建消息
    rebuilt_msg = Message.from_dict(msg_dict)
    print("\n重建的消息:")
    print(f"Role: {rebuilt_msg.role}")
    print(f"Action: {rebuilt_msg.action}")
    print(f"Content: {rebuilt_msg.content}")
    print(f"Metadata: {rebuilt_msg.metadata}")