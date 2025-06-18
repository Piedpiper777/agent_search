# 此文件定义了搜索工具的基类，提供了通用的搜索逻辑和结果处理方法
from typing import Dict, Any, List, Optional
from ...core.custom_types import SearchResult, ToolResult
from ..base_tool import BaseTool
import time

class SearchBaseTool(BaseTool):
    """搜索工具基类，定义所有搜索工具的通用接口，继承自 BaseTool"""
    
    def _setup(self) -> None:
        """初始化搜索工具的基本配置"""
        self.required_fields = ["query"]
        self.optional_fields = [
            "limit",          # 返回结果数量限制
            "threshold",      # 相关性阈值
            "filters"         # 过滤条件
        ]
        
    def _format_results(self, results: List[Dict[str, Any]], source: str) -> SearchResult:
        """格式化搜索结果"""
        return {
            "facts": results,
            "confidence": self._calculate_confidence(results),
            "source": source,
            "metadata": {
                "result_count": len(results),
                "timestamp": time.time()
            }
        }
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """计算结果的置信度"""
        if not results:
            return 0.0
        # 默认实现：根据结果数量计算基础置信度
        # 这里假设结果数量越多，置信度越高，最多5个结果时达到最大置信度1.0
        base_confidence = min(len(results) / 5, 1.0)  # 最多5个结果时达到最大置信度
        return base_confidence
    
    def _validate_results(self, results: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """验证和过滤搜索结果"""
        """"只有相关性得分高于阈值的结果才会被保留"""
        return [
            result for result in results
            if result.get("relevance", 0) >= threshold #relevance应由search方法返回的结果包含
        ]
    
    def _preprocess_query(self, query: str) -> str:
        """预处理查询字符串"""
        # 基础的查询预处理，子类可以重写此方法实现更复杂的处理
        return query.strip() # 去除首尾空格
    
    def _execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """执行搜索的核心逻辑"""
        try:
            # 获取和处理输入参数
            query = self._preprocess_query(input_data["query"])
            limit = input_data.get("limit", 10) # 返回结果数量限制，默认10
            threshold = input_data.get("threshold", 0.5) # 相关性阈值，默认0.5
            filters = input_data.get("filters", {}) # 过滤条件，默认空字典
            
            # 执行具体的搜索逻辑（子类必须实现）
            results = self._search(query, limit, filters)
            
            # 验证和过滤结果
            validated_results = self._validate_results(results, threshold)
            
            # 格式化结果
            search_result = self._format_results(
                validated_results, 
                source=self.__class__.__name__
            )
            
            return {
                "success": True,
                "data": search_result,
                "error": None,
                "metadata": {
                    "original_query": query,
                    "processed_query": query,
                    "limit": limit,
                    "threshold": threshold,
                    "filters": filters
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "metadata": {
                    "error_type": type(e).__name__,
                    "original_query": input_data.get("query", "")
                }
            }
    
    def _search(self, query: str, limit: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """具体的搜索实现（子类必须重写此方法）"""
        raise NotImplementedError("搜索方法必须由子类实现")

# 使用示例
if __name__ == "__main__":
    # 创建一个简单的测试搜索工具
    class DummySearchTool(SearchBaseTool):
        def _search(self, query: str, limit: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
            # 模拟搜索结果
            return [
                {"content": f"Result {i}", "relevance": 0.8 - (i * 0.1)}
                for i in range(limit)
            ]
    
    # 测试搜索工具
    search_tool = DummySearchTool(
        name="dummy_search",
        description="用于测试的搜索工具"
    )
    
    # 执行测试查询
    result = search_tool.execute({
        "query": "test query",
        "limit": 3,
        "threshold": 0.6
    })
    
    print("\n搜索结果:")
    print(result)