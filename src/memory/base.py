"""
记忆系统基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

class MemoryBase(ABC):
    """记忆系统基类"""
    
    @abstractmethod
    def save(self, key: str, value: Any) -> None:
        """保存记忆"""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Any:
        """加载记忆"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清除所有记忆"""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """获取记忆大小"""
        pass