"""
长期记忆实现
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from .base import MemoryBase

class LongTermMemory(MemoryBase):
    """长期记忆实现，基于向量数据库"""
    
    def __init__(self, api_key: str, base_url: str, user_name: str,
                 model: str = "text-embedding-3-small",
                 collection_name_prefix: str = "memory_",
                 persist_directory_prefix: str = "./chroma_db_"):
        """
        初始化长期记忆
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            user_name: 用户名
            model: 嵌入模型名称
            collection_name_prefix: 集合名称前缀
            persist_directory_prefix: 持久化目录前缀
        """
        self.user_name = user_name
        self.collection_name = f"{collection_name_prefix}{user_name}"
        self.persist_directory = f"{persist_directory_prefix}{user_name}"
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url
        )
        
        # 初始化向量数据库
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def save(self, key: str, value: Any) -> None:
        """
        保存记忆
        
        Args:
            key: 记忆键名
            value: 记忆值，应该是包含user_input、assistant_response和extracted_info的字典
        """
        if key == "memory" and isinstance(value, dict):
            user_input = value.get("user_input", "")
            assistant_response = value.get("assistant_response", "")
            extracted_info = value.get("extracted_info", {})
            
            timestamp = datetime.now().isoformat()
            
            # 创建文档内容
            if extracted_info:
                doc_content = f"时间: {timestamp}\n用户说: {user_input}\n提取的信息: {json.dumps(extracted_info, ensure_ascii=False)}"
                doc_type = "conversation_with_extraction"
            else:
                doc_content = f"时间: {timestamp}\n用户: {user_input}\n助手: {assistant_response}"
                doc_type = "conversation"
            
            # 创建文档对象
            doc = Document(
                page_content=doc_content,
                metadata={
                    "timestamp": timestamp,
                    "user_input": user_input,
                    "type": doc_type
                }
            )
            
            # 添加到向量数据库
            self.vector_store.add_documents([doc])
    
    def load(self, key: str, **kwargs) -> Any:
        """
        加载记忆
        
        Args:
            key: 记忆键名
            **kwargs: 额外参数，如k=3表示检索3条相关记忆
            
        Returns:
            相关记忆列表
        """
        if key == "relevant_memories":
            query = kwargs.get("query", "")
            k = kwargs.get("k", 3)
            
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                if docs:
                    memories = "\n".join([f"- {doc.page_content}" for doc in docs])
                    return memories
                return "暂无相关历史记忆"
            except Exception as e:
                print(f"记忆检索出错: {e}")
                return "暂无相关历史记忆"
        return None
    
    def clear(self) -> None:
        """清除所有记忆"""
        # Chroma没有直接清除集合的API，我们删除持久化目录
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
    
    def get_size(self) -> int:
        """
        获取记忆大小
        
        Returns:
            记忆数量
        """
        try:
            # 获取集合中的文档数量
            return self.vector_store._collection.count()
        except Exception as e:
            print(f"获取记忆大小出错: {e}")
            return 0