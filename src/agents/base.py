"""
Agent基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from src.config import config
from src.prompts import PromptManager

class BaseAgent(ABC):
    """
    Agent基类，定义所有Agent的核心接口
    """
    
    def __init__(self, user_name: str = None):
        """
        初始化Agent
        
        Args:
            user_name: 用户名，默认使用配置中的默认用户名
        """
        self.user_name = user_name or config.user_config["default_user_name"]
        self.prompt_manager = PromptManager()
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            **config.llm_config
        )
    
    @abstractmethod
    def chat(self, user_input: str) -> str:
        """
        与用户对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        """
        清除所有记忆
        """
        pass
    
    @abstractmethod
    def get_profile(self) -> Dict[str, Any]:
        """
        获取用户画像
        
        Returns:
            用户画像字典
        """
        pass
    
    @abstractmethod
    def show_history(self) -> None:
        """
        显示对话历史
        """
        pass
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        设置系统提示词
        
        Args:
            prompt: 系统提示词
        """
        # 默认实现，子类可以重写
        self.prompt_manager.update_template(
            category="system",
            template_name="default",
            template=prompt
        )