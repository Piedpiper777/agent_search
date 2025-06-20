from typing import Dict, Type, Optional
from ...tools.base_tool import BaseTool
from ...tools.decompose import DecomposeTool
from ...tools.search.search_kg import KGSearchTool
from ...tools.search.search_vector import VectorSearchTool
from ...tools.generate import GenerateAnswerTool

class ToolRegistry:
    """工具注册表，管理所有可用的工具"""
    
    _tools: Dict[str, Type[BaseTool]] = {
        "decompose": DecomposeTool,
        "search_kg": KGSearchTool,
        "search_vector": VectorSearchTool,
        "generate": GenerateAnswerTool
    }
    
    @classmethod
    def register(cls, name: str, tool_class: Type[BaseTool]) -> None:
        """注册新工具"""
        if name in cls._tools:
            raise ValueError(f"工具 {name} 已存在")
        cls._tools[name] = tool_class
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[Type[BaseTool]]:
        """获取工具类"""
        return cls._tools.get(name)
    
    @classmethod
    def create_tool(cls, name: str, **kwargs) -> Optional[BaseTool]:
        """创建工具实例"""
        tool_class = cls.get_tool(name)
        if tool_class:
            return tool_class(name=name, **kwargs)
        return None
    
    @classmethod
    def list_tools(cls) -> Dict[str, Type[BaseTool]]:
        """列出所有已注册的工具"""
        return cls._tools.copy()

# 使用示例
if __name__ == "__main__":
    # 列出所有工具
    print("\n已注册的工具:")
    for name, tool_class in ToolRegistry.list_tools().items():
        print(f"- {name}: {tool_class.__doc__}")
    
    # 创建工具实例
    decompose_tool = ToolRegistry.create_tool(
        "decompose",
        description="查询分解工具"
    )
    print(f"\n创建工具: {decompose_tool.name}")