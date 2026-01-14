"""
配置管理模块
统一加载和管理所有配置
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """配置管理类"""
    
    def __init__(self):
        """初始化配置"""
        # 加载.env文件
        load_dotenv()
        
        # 基本配置
        self.api_key = self._get_env("API_KEY", required=True)
        self.base_url = self._get_env("BASE_URL", default="https://api.deepseek.com")
        
        # LLM配置
        self.llm_config = {
            "model": self._get_env("LLM_MODEL", default="deepseek-chat"),
            "temperature": float(self._get_env("LLM_TEMPERATURE", default="0.7")),
            "max_tokens": int(self._get_env("LLM_MAX_TOKENS", default="4096")),
        }
        
        # 嵌入模型配置
        self.embedding_config = {
            "model": self._get_env("EMBEDDING_MODEL", default="text-embedding-3-small"),
        }
        
        # 记忆配置
        self.memory_config = {
            "short_term": {
                "memory_key": "chat_history",
                "return_messages": True
            },
            "long_term": {
                "collection_name_prefix": "memory_",
                "persist_directory_prefix": "./chroma_db_"
            }
        }
        
        # 用户配置
        self.user_config = {
            "default_user_name": self._get_env("DEFAULT_USER_NAME", default="chenkx")
        }
    
    def _get_env(self, key: str, default: str = None, required: bool = False) -> str:
        """
        从环境变量获取值
        
        Args:
            key: 环境变量名
            default: 默认值
            required: 是否必填
            
        Returns:
            环境变量值
            
        Raises:
            ValueError: 当必填项未设置时
        """
        value = os.environ.get(key, default)
        if required and value is None:
            raise ValueError(f"配置项 {key} 未设置！请检查.env文件或环境变量")
        return value
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.llm_config
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取嵌入模型配置"""
        return self.embedding_config
    
    def get_memory_config(self) -> Dict[str, Any]:
        """获取记忆配置"""
        return self.memory_config
    
    def get_user_config(self) -> Dict[str, Any]:
        """获取用户配置"""
        return self.user_config

# 创建全局配置实例
config = Config()