# 此脚本用于将文档数据导入到Milvus向量数据库中
from typing import List, Dict
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from agent_project.config.vector_db import VECTOR_DB_CONFIG

def create_collection(dimension: int) -> Collection:
    """创建Milvus集合"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
    ]
    schema = CollectionSchema(fields=fields, description="document collection")
    collection = Collection(name="documents", schema=schema)
    
    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

def import_documents(file_path: str):
    """导入文档数据"""
    # 初始化编码器
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = 384
    
    # 连接数据库
    connections.connect(**VECTOR_DB_CONFIG["connection"])
    
    # 创建集合
    collection = create_collection(dimension)
    
    # 读取文档
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # 批量处理文档
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # 生成向量
        contents = [doc["content"] for doc in batch]
        embeddings = encoder.encode(contents)
        
        # 准备插入数据
        entities = [
            [i + j for j in range(len(batch))],  # id
            contents,  # content
            embeddings.tolist()  # embedding
        ]
        
        # 插入数据
        collection.insert(entities)
        print(f"已处理 {i + len(batch)} 条文档")
    
    # 创建索引
    collection.flush()
    print("数据导入完成")

if __name__ == "__main__":
    import_documents("path/to/your/documents.json")
    #可以把三元组数据转换为JSON格式，存储在documents.json文件中