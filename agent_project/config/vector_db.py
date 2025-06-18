# 此文件定义了向量数据库的配置
from pathlib import Path

VECTOR_DB_CONFIG = {
    "dimension": 384,  # 向量维度
    "index_path": str(Path(__file__).parent.parent / "data" / "vector_indices"),# 目前还没有建立
    "collection_name": "documents",
    "connection": {
        "host": "localhost",
        "port": 19530,  # Milvus默认端口
    }
}