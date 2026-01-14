"""
具有记忆功能的Agent
"""

import json
from typing import Dict, Any
from langchain.chains import LLMChain
from src.config import config
from src.memory import ShortTermMemory, LongTermMemory, UserProfile
from .base import BaseAgent

class MemoryAgent(BaseAgent):
    """
    具有长期记忆和用户画像管理功能的Agent
    """
    
    def __init__(self, user_name: str = None):
        """
        初始化MemoryAgent
        
        Args:
            user_name: 用户名
        """
        super().__init__(user_name)
        
        # 初始化记忆系统
        self.short_term_memory = ShortTermMemory(**config.memory_config["short_term"])
        
        self.long_term_memory = LongTermMemory(
            api_key=config.api_key,
            base_url=config.base_url,
            user_name=self.user_name,
            **config.memory_config["long_term"]
        )
        
        # 初始化用户画像
        self.user_profile = UserProfile(self.user_name)
    
    def chat(self, user_input: str) -> str:
        """
        与用户对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        # 1. 检索相关记忆
        relevant_memories = self.long_term_memory.load(
            key="relevant_memories",
            query=user_input,
            k=3
        )
        
        # 2. 格式化用户画像
        profile_summary = self.user_profile.to_string()
        
        # 3. 生成回复
        conversation_prompt = self.prompt_manager.get_system_prompt("memory_aware")
        messages = conversation_prompt.format_messages(
            user_name=self.user_name,
            user_profile=profile_summary,
            relevant_memories=relevant_memories,
            input=user_input
        )
        
        response = self.llm.invoke(messages)
        
        # 4. 更新短期记忆
        self.short_term_memory.save(
            key="context",
            value={
                "input": {"input": user_input},
                "output": {"output": response.content}
            }
        )
        
        # 5. 提取并存储长期记忆
        self._extract_and_store_memory(user_input, response.content)
        
        return response.content
    
    def _extract_and_store_memory(self, user_input: str, assistant_response: str) -> None:
        """
        从对话中提取并存储长期记忆
        
        Args:
            user_input: 用户输入
            assistant_response: 助手回复
        """
        try:
            # 使用LLM提取信息
            extraction_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_manager.get_extraction_prompt()
            )
            
            conversation = f"用户: {user_input}\n助手: {assistant_response}"
            result = extraction_chain.run(
                conversation=conversation,
                user_name=self.user_name
            )
            
            # 解析JSON结果
            extracted_info = json.loads(result)
            
            # 更新用户画像
            self.user_profile.update(extracted_info)
            
            # 存储到长期记忆
            self.long_term_memory.save(
                key="memory",
                value={
                    "user_input": user_input,
                    "assistant_response": assistant_response,
                    "extracted_info": extracted_info
                }
            )
            
        except Exception as e:
            print(f"信息提取出错: {e}")
            # 即使提取失败，也存储原始对话
            self.long_term_memory.save(
                key="memory",
                value={
                    "user_input": user_input,
                    "assistant_response": assistant_response,
                    "extracted_info": {}
                }
            )
    
    def clear_memory(self) -> None:
        """
        清除所有记忆
        """
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.user_profile.clear()
        print("记忆已清除")
    
    def get_profile(self) -> Dict[str, Any]:
        """
        获取用户画像
        
        Returns:
            用户画像字典
        """
        return self.user_profile.get_profile()
    
    def show_history(self) -> None:
        """
        显示对话历史
        """
        chat_history = self.short_term_memory.load(key="chat_history")
        if not chat_history:
            print("暂无对话历史")
            return
        
        for i, msg in enumerate(chat_history):
            role = "用户" if hasattr(msg, "type") and msg.type == "human" else "AI"
            print(f"\n[{role}]: {msg.content}")