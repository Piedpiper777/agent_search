"""定义了项目中使用的自定义类型和别名"""
from typing import Dict, List, Any, Union, TypeVar, Callable, TypedDict, Optional

# 基础类型别名
JSON = Dict[str, Any]
MetaData = Dict[str, Any]

# 工具相关类型
class ToolResult(TypedDict):
    """工具执行结果类型，包含成功标志、数据、错误信息和元数据，success表示工具执行是否成功，data是工具返回的数据，error是错误信息，metadata是元数据"""
    success: bool
    data: Any
    error: Optional[str]
    metadata: MetaData

class ToolConfig(TypedDict):
    """工具配置类型，包含工具名称、描述、必需字段和可选字段，name表示工具的名称，description是工具的描述，required_fields是必需字段列表，optional_fields是可选字段列表"""
    name: str
    description: str
    required_fields: List[str] 
    optional_fields: List[str]

ToolFunction = Callable[[Any], ToolResult]
"""工具函数类型，接受任意输入，返回ToolResult类型的结果"""

# 搜索相关类型
class SearchResult(TypedDict):
    """搜索结果类型，包含搜索到的内容、置信度、来源和元数据，facts是搜索到的内容列表，confidence是搜索结果的置信度，source是数据来源，metadata是附加信息"""
    facts: List[Dict[str, Any]]
    confidence: float
    source: str
    metadata: MetaData

# LLM相关类型
class LLMResponse(TypedDict):
    """LLM响应类型，包含生成的文本、置信度和元数据，text是生成的文本内容，confidence是生成的置信度，metadata是附加信息"""
    text: str
    confidence: float
    metadata: MetaData

# 状态相关类型
StateData = TypeVar('StateData')  # 用于泛型状态数据
"""状态数据类型，允许任何类型的状态数据"""

class ValidationResult(TypedDict):
    """验证结果类型，包含验证是否成功、错误信息列表"""
    valid: bool
    errors: List[str]