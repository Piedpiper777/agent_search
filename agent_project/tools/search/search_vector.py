# 此文件是向量搜索工具的实现，使用Milvus进行高效相似度检索
# 但是向量数据库还没有建立
from typing import Dict, Any, List
import numpy as np
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from .base_search import SearchBaseTool 
from ...config.vector_db import VECTOR_DB_CONFIG
import time

class VectorSearchTool(SearchBaseTool):
    """向量搜索工具，使用Milvus进行高效相似度检索"""
    
    def _setup(self) -> None:
        """初始化向量搜索工具"""
        super()._setup()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2') # 修改为本地加载，只需将模型名称改为本地模型路径
        self._connect_db()
        self.collection = self._get_collection()
    
    def _connect_db(self) -> None:
        """连接向量数据库"""
        try:
            connections.connect(
                alias="default",
                **VECTOR_DB_CONFIG["connection"]
            )
        except Exception as e:
            raise RuntimeError(f"连接向量数据库失败: {str(e)}")
    
    def _get_collection(self) -> Collection:
        """获取或创建集合"""
        collection_name = VECTOR_DB_CONFIG["collection_name"]
        
        if not utility.exists_collection(collection_name):
            raise RuntimeError(f"集合 {collection_name} 不存在")
            
        collection = Collection(collection_name)
        collection.load()
        return collection
    
    def _search(self, query: str, limit: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行向量搜索"""
        try:
            # 对查询文本进行编码
            query_vector = self.encoder.encode([query])[0].tolist()
            
            # 构建搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            
            # 执行搜索
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",  # 向量字段名
                param=search_params,
                limit=limit,
                expr=self._build_filter_expr(filters)  # 构建过滤表达式
            )
            
            # 整理搜索结果
            processed_results = []
            for hits in results:
                for hit in hits:
                    similarity = 1 / (1 + hit.distance)  # 转换距离为相似度
                    doc = {
                        "id": hit.id,
                        "content": hit.entity.get("content"),
                        "relevance": float(similarity),
                        "distance": float(hit.distance)
                    }
                    processed_results.append(doc)
            
            return processed_results
            
        except Exception as e:
            raise RuntimeError(f"向量搜索失败: {str(e)}")
    
    def _build_filter_expr(self, filters: Dict[str, Any]) -> str:
        """构建过滤表达式"""
        if not filters:
            return ""
        # TODO: 实现过滤表达式构建逻辑，需后续补充
        return ""

# 使用示例
if __name__ == "__main__":
    # 创建向量搜索工具实例
    vector_search = VectorSearchTool(
        name="vector_search",
        description="使用向量相似度进行文档检索"
    )
    
    # 执行测试查询
    result = vector_search.execute({
        "query": "诺兰的电影inception",
        "limit": 2,
        "threshold": 0.5
    })
    
    print("\n向量搜索结果:")
    if result["success"]:
        for fact in result["data"]["facts"]:
            print(f"\n- 内容: {fact['content']}")
            print(f"  相关性: {fact['relevance']:.3f}")
            print(f"  排名: {fact['rank']}")
    else:
        print(f"搜索失败: {result['error']}")