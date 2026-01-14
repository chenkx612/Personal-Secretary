"""
提示词管理模块
"""

from typing import Dict, Any
from langchain.prompts import PromptTemplate, ChatPromptTemplate

class PromptManager:
    """提示词管理类"""
    
    def __init__(self):
        """初始化提示词模板"""
        self.templates = {
            "system": {
                "default": "你是 {user_name} 的 AI 个人助手，请使用简洁且专业的回复风格。",
                "memory_aware": """
你是{user_name}的个人 AI 助手，具有长期记忆能力。

关于{user_name}的已知信息：
{user_profile}

相关历史记忆：
{relevant_memories}

请基于这些信息，提供个性化、贴心的回复。如果发现用户提到的新信息，自然地融入对话。
                """
            },
            "extraction": {
                "default": """
从以下对话中提取关于{user_name}的重要信息。

对话内容：
{conversation}

请以 JSON 格式返回提取的信息，包括以下类别（如果有）：
- personal_info: 个人基本信息（姓名、年龄、职业等）
- interests: 兴趣爱好
- preferences: 偏好（喜欢/不喜欢的事物）
- goals: 目标和计划
- experiences: 重要经历和事件
- relationships: 人际关系
- habits: 生活习惯
- concerns: 关注的问题

只返回 JSON 格式，不要包含其他文字：
                """
            }
        }
        
        # 编译后的提示词模板
        self.compiled_templates = self._compile_templates()
    
    def _compile_templates(self) -> Dict[str, Any]:
        """
        编译所有提示词模板
        
        Returns:
            编译后的提示词模板字典
        """
        compiled = {
            "system": {},
            "extraction": {}
        }
        
        # 编译系统提示词
        compiled["system"]["default"] = PromptTemplate(
            input_variables=["user_name"],
            template=self.templates["system"]["default"]
        )
        
        compiled["system"]["memory_aware"] = ChatPromptTemplate.from_messages([
            ("system", self.templates["system"]["memory_aware"])
        ])
        
        # 编译信息提取提示词
        compiled["extraction"]["default"] = PromptTemplate(
            input_variables=["conversation", "user_name"],
            template=self.templates["extraction"]["default"]
        )
        
        return compiled
    
    def get_system_prompt(self, template_name: str = "default") -> Any:
        """
        获取系统提示词模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            系统提示词模板
        """
        return self.compiled_templates["system"].get(template_name)
    
    def get_extraction_prompt(self, template_name: str = "default") -> Any:
        """
        获取信息提取提示词模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            信息提取提示词模板
        """
        return self.compiled_templates["extraction"].get(template_name)
    
    def update_template(self, category: str, template_name: str, template: str) -> None:
        """
        更新提示词模板
        
        Args:
            category: 模板类别（system/extraction）
            template_name: 模板名称
            template: 新的模板内容
        """
        if category in self.templates and template_name in self.templates[category]:
            self.templates[category][template_name] = template
            # 重新编译模板
            self.compiled_templates = self._compile_templates()
    
    def add_template(self, category: str, template_name: str, template: str) -> None:
        """
        添加新的提示词模板
        
        Args:
            category: 模板类别（system/extraction）
            template_name: 模板名称
            template: 模板内容
        """
        if category not in self.templates:
            self.templates[category] = {}
        
        self.templates[category][template_name] = template
        # 重新编译模板
        self.compiled_templates = self._compile_templates()