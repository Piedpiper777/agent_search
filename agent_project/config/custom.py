import os
from typing import Dict, Any
from .default import DEFAULT_CONFIG

# 从环境变量加载配置
def load_env_config() -> Dict[str, Any]:
    """从环境变量加载配置"""
    return {
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", DEFAULT_CONFIG["llm"]["provider"]),
            "model": os.getenv("LLM_MODEL", DEFAULT_CONFIG["llm"]["model"]),
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
        },
        "vector_db": {
            "host": os.getenv("VECTOR_DB_HOST", DEFAULT_CONFIG["vector_db"]["host"]),
            "password": os.getenv("VECTOR_DB_PASSWORD", ""),
        },
        "knowledge_graph": {
            "uri": os.getenv("KG_URI", DEFAULT_CONFIG["knowledge_graph"]["uri"]),
            "password": os.getenv("KG_PASSWORD", DEFAULT_CONFIG["knowledge_graph"]["password"]),
        }
    }

# 自定义配置
CUSTOM_CONFIG = {
    "llm": {
        "temperature": 0.8,  # 覆盖默认温度
    },
    "tools": {
        "search_vector": {
            "top_k": 10,  # 覆盖默认检索数量
        }
    },
    "logging": {
        "level": "DEBUG"  # 覆盖默认日志级别
    }
}

# 合并配置
def get_config() -> Dict[str, Any]:
    """获取最终配置"""
    config = DEFAULT_CONFIG.copy()
    
    # 合并环境变量配置
    env_config = load_env_config()
    _deep_update(config, env_config)
    
    # 合并自定义配置
    _deep_update(config, CUSTOM_CONFIG)
    
    return config

def _deep_update(base: Dict, update: Dict) -> None:
    """递归更新字典"""
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _deep_update(base[key], value)
        else:
            base[key] = value