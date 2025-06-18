from typing import Dict, Any

DEFAULT_CONFIG = {
    # LLM配置
    "llm": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 2000,
        "timeout": 30,
    },
    
    # 向量数据库配置
    "vector_db": {
        "host": "localhost",
        "port": 19530,
        "collection": "documents",
        "dimension": 384,
        "index_type": "IVF_FLAT",
        "metric_type": "L2"
    },
    
    # 知识图谱配置
    "knowledge_graph": {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
        "database": "neo4j"
    },
    
    # 工具配置
    "tools": {
        "decompose": {
            "enabled": True,
            "max_subqueries": 3
        },
        "search_kg": {
            "enabled": True,
            "max_hops": 2,
            "min_confidence": 0.6
        },
        "search_vector": {
            "enabled": True,
            "top_k": 5,
            "threshold": 0.7
        },
        "generate": {
            "enabled": True,
            "max_length": 1000
        }
    },
    
    # 控制器配置
    "controller": {
        "type": "llm",  # or "rule"
        "max_steps": 10,
        "timeout": 300
    },
    
    # 日志配置
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent.log"
    }
}