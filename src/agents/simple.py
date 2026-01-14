"""
简单对话Agent
"""

from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from .base import BaseAgent

class SimpleAgent(BaseAgent):
    """
    简单对话Agent，具有基本的对话功能
    """
    
    def __init__(self, user_name: str = None):
        """
        初始化SimpleAgent
        
        Args:
            user_name: 用户名
        """
        super().__init__(user_name)
        self.conversation_history = []
    
    def chat(self, user_input: str) -> str:
        """
        与用户对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        # 构建消息列表
        messages = [SystemMessage(content=self.prompt_manager.templates["system"]["default"].format(user_name=self.user_name))]
        messages.extend(self.conversation_history)
        messages.append(HumanMessage(content=user_input))
        
        # 获取回复
        response = self.llm.invoke(messages)
        
        # 保存对话历史
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=response.content))
        
        return response.content
    
    def clear_memory(self) -> None:
        """
        清除所有记忆
        """
        self.conversation_history = []
        print("对话历史已清空")
    
    def get_profile(self) -> Dict[str, Any]:
        """
        获取用户画像
        
        Returns:
            空字典，SimpleAgent不支持用户画像
        """
        return {}
    
    def show_history(self) -> None:
        """
        显示对话历史
        """
        if not self.conversation_history:
            print("暂无对话历史")
            return
        
        for i, msg in enumerate(self.conversation_history):
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            print(f"\n[{role}]: {msg.content}")