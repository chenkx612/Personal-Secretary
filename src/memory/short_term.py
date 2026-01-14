"""
短期记忆实现
"""

from typing import Any, Dict, List
from langchain.memory import ConversationBufferMemory
from .base import MemoryBase

class ShortTermMemory(MemoryBase):
    """短期记忆实现，基于对话历史"""
    
    def __init__(self, memory_key: str = "chat_history", return_messages: bool = True):
        """
        初始化短期记忆
        
        Args:
            memory_key: 记忆键名
            return_messages: 是否返回消息对象
        """
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages
        )
    
    def save(self, key: str, value: Any) -> None:
        """
        保存记忆
        
        Args:
            key: 记忆键名
            value: 记忆值
        """
        if key == "context":
            # 保存对话上下文
            if isinstance(value, dict) and "input" in value and "output" in value:
                self.memory.save_context(value["input"], value["output"])
    
    def load(self, key: str) -> Any:
        """
        加载记忆
        
        Args:
            key: 记忆键名
            
        Returns:
            记忆值
        """
        if key == "chat_history":
            return self.memory.load_memory_variables({}).get("chat_history", [])
        return None
    
    def clear(self) -> None:
        """清除所有记忆"""
        self.memory.clear()
    
    def get_size(self) -> int:
        """获取记忆大小"""
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        return len(chat_history) if chat_history else 0