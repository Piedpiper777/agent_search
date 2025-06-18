"""定义了工具基类，所有具体工具都应继承自此类"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from ..core.custom_types import ToolResult
from ..core.message import Message, MessageRole
import time

class BaseTool(ABC):
    """工具基类，定义所有工具必须实现的接口"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """初始化工具特定的配置"""
        pass
    
    @abstractmethod
    def _execute(self, message: Message) -> Message:
        """执行工具的核心逻辑"""
        pass
    
    def validate_message(self, message: Message) -> bool:
        """验证输入消息是否满足要求"""
        if not isinstance(message, Message):
            return False
        return True
    
    def execute(self, message: Message) -> Message:
        """执行工具，包含输入验证和错误处理"""
        try:
            # 验证消息
            if not self.validate_message(message):
                return Message(
                    role=MessageRole.ERROR,
                    action=None,
                    content={
                        "error": "Invalid message format"
                    },
                    metadata={
                        "tool_name": self.name,
                        "error_type": "ValidationError"
                    }
                )
            
            # 执行核心逻辑
            result = self._execute(message)
            
            # 确保返回的是Message对象
            if not isinstance(result, Message):
                raise TypeError("Tool must return a Message object")
            
            return result
            
        except Exception as e:
            # 统一的错误处理
            return Message(
                role=MessageRole.ERROR,
                action=None,
                content={
                    "error": str(e)
                },
                metadata={
                    "tool_name": self.name,
                    "error_type": type(e).__name__,
                    "timestamp": time.time()
                },
                parent_id=message.query_id
            )

# 使用示例
if __name__ == "__main__":
    # 创建一个简单的测试工具
    class EchoTool(BaseTool):
        def _setup(self) -> None:
            self.required_fields = ["message"]
            self.optional_fields = ["prefix"]
            
        def _execute(self, input_data: Message) -> Message:
            prefix = input_data.get("prefix", "")
            message = input_data["message"]
            return Message(
                role=MessageRole.INFO,
                action="echo",
                content={
                    "message": f"{prefix}{message}"
                },
                metadata={
                    "message_length": len(message)
                }
            )
    
    # 测试工具
    echo_tool = EchoTool(
        name="echo",
        description="简单的消息回显工具"
    )
    
    # 测试有效输入
    result = echo_tool.execute(Message(
        role=MessageRole.USER,
        action="send_message",
        content={
            "message": "Hello, World!",
            "prefix": "Echo: "
        }
    ))
    print("\n有效输入测试:")
    print(result)
    
    # 测试无效输入
    result = echo_tool.execute(Message(
        role=MessageRole.USER,
        action="send_message",
        content={
            "prefix": "Echo: "  # 缺少必需的 message 字段
        }
    ))
    print("\n无效输入测试:")
    print(result)