"""
具有自主学习和长期记忆管理功能
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class PersonalMemoryAgent:
    """具有长期记忆管理能力的个人助手Agent"""
    
    def __init__(self, deepseek_api_key: str, user_name: str = "用户"):
        """
        初始化Agent
        
        Args:
            deepseek_api_key: DeepSeek API密钥
            user_name: 用户名称
        """
        self.user_name = user_name
        self.profile_file = f"user_profile_{user_name}.json"
        
        # 配置DeepSeek API (使用OpenAI兼容接口)
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7
        )
        
        # 短期记忆：对话历史
        self.short_term_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 长期记忆：向量数据库
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com"
        )
        
        self.vector_store = Chroma(
            collection_name=f"memory_{user_name}",
            embedding_function=self.embeddings,
            persist_directory=f"./chroma_db_{user_name}"
        )
        
        # 结构化用户画像
        self.user_profile = self._load_profile()
        
        # 提示词模板
        self._setup_prompts()
    
    def _setup_prompts(self):
        """设置提示词模板"""
        
        # 信息提取提示词
        self.extraction_prompt = PromptTemplate(
            input_variables=["conversation", "user_name"],
            template="""
从以下对话中提取关于{user_name}的重要信息。

对话内容：
{conversation}

请以JSON格式返回提取的信息，包括以下类别（如果有）：
- personal_info: 个人基本信息（姓名、年龄、职业等）
- interests: 兴趣爱好
- preferences: 偏好（喜欢/不喜欢的事物）
- goals: 目标和计划
- experiences: 重要经历和事件
- relationships: 人际关系
- habits: 生活习惯
- concerns: 关注的问题

只返回JSON格式，不要包含其他文字：
"""
        )
        
        # 对话生成提示词
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是{user_name}的个人AI助手，具有长期记忆能力。

关于{user_name}的已知信息：
{user_profile}

相关历史记忆：
{relevant_memories}

请基于这些信息，提供个性化、贴心的回复。如果发现用户提到的新信息，自然地融入对话。"""),
            ("human", "{input}")
        ])
    
    def _load_profile(self) -> Dict:
        """加载用户画像"""
        if os.path.exists(self.profile_file):
            with open(self.profile_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "personal_info": {},
            "interests": [],
            "preferences": {"likes": [], "dislikes": []},
            "goals": [],
            "experiences": [],
            "relationships": [],
            "habits": [],
            "concerns": []
        }
    
    def _save_profile(self):
        """保存用户画像"""
        with open(self.profile_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
    
    def _retrieve_relevant_memories(self, query: str, k: int = 3) -> str:
        """从长期记忆中检索相关信息"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            if docs:
                memories = "\n".join([f"- {doc.page_content}" for doc in docs])
                return memories
            return "暂无相关历史记忆"
        except:
            return "暂无相关历史记忆"
    
    def _extract_and_store_info(self, user_input: str, assistant_response: str):
        """从对话中提取并存储信息"""
        conversation = f"用户: {user_input}\n助手: {assistant_response}"
        
        # 使用LLM提取信息
        extraction_chain = LLMChain(
            llm=self.llm,
            prompt=self.extraction_prompt
        )
        
        try:
            result = extraction_chain.run(
                conversation=conversation,
                user_name=self.user_name
            )
            
            # 解析JSON结果
            extracted_info = json.loads(result)
            
            # 更新用户画像
            self._update_profile(extracted_info)
            
            # 存储到向量数据库
            self._store_to_vector_db(user_input, assistant_response, extracted_info)
            
        except Exception as e:
            print(f"信息提取出错: {e}")
            # 即使提取失败，也存储原始对话
            self._store_conversation_only(user_input, assistant_response)
    
    def _update_profile(self, extracted_info: Dict):
        """更新用户画像"""
        for category, value in extracted_info.items():
            if category in self.user_profile:
                if isinstance(self.user_profile[category], dict):
                    self.user_profile[category].update(value)
                elif isinstance(self.user_profile[category], list):
                    if isinstance(value, list):
                        for item in value:
                            if item not in self.user_profile[category]:
                                self.user_profile[category].append(item)
                    else:
                        if value not in self.user_profile[category]:
                            self.user_profile[category].append(value)
        
        self._save_profile()
    
    def _store_to_vector_db(self, user_input: str, assistant_response: str, extracted_info: Dict):
        """存储信息到向量数据库"""
        timestamp = datetime.now().isoformat()
        
        # 创建文档
        doc_content = f"时间: {timestamp}\n用户说: {user_input}\n提取的信息: {json.dumps(extracted_info, ensure_ascii=False)}"
        
        doc = Document(
            page_content=doc_content,
            metadata={
                "timestamp": timestamp,
                "user_input": user_input,
                "type": "conversation_with_extraction"
            }
        )
        
        self.vector_store.add_documents([doc])
    
    def _store_conversation_only(self, user_input: str, assistant_response: str):
        """仅存储对话内容"""
        timestamp = datetime.now().isoformat()
        doc_content = f"时间: {timestamp}\n用户: {user_input}\n助手: {assistant_response}"
        
        doc = Document(
            page_content=doc_content,
            metadata={
                "timestamp": timestamp,
                "type": "conversation"
            }
        )
        
        self.vector_store.add_documents([doc])
    
    def chat(self, user_input: str) -> str:
        """
        与用户对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        # 1. 检索相关记忆
        relevant_memories = self._retrieve_relevant_memories(user_input)
        
        # 2. 格式化用户画像
        profile_summary = json.dumps(self.user_profile, ensure_ascii=False, indent=2)
        
        # 3. 生成回复
        conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.conversation_prompt
        )
        
        response = conversation_chain.run(
            user_name=self.user_name,
            user_profile=profile_summary,
            relevant_memories=relevant_memories,
            input=user_input
        )
        
        # 4. 更新短期记忆
        self.short_term_memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        # 5. 提取并存储长期记忆（异步进行，不阻塞响应）
        try:
            self._extract_and_store_info(user_input, response)
        except Exception as e:
            print(f"记忆存储失败: {e}")
        
        return response
    
    def show_profile(self) -> Dict:
        """显示当前用户画像"""
        return self.user_profile
    
    def clear_memory(self):
        """清除所有记忆"""
        self.short_term_memory.clear()
        if os.path.exists(self.profile_file):
            os.remove(self.profile_file)
        print("记忆已清除")


# 使用示例
def main():
    """主函数示例"""
    
    # 配置API密钥
    DEEPSEEK_API_KEY = "your-deepseek-api-key"  # 替换为你的API密钥
    
    # 创建Agent
    agent = PersonalMemoryAgent(
        deepseek_api_key=DEEPSEEK_API_KEY,
        user_name="张三"
    )
    
    print("🤖 个人记忆Agent已启动！")
    print("我会记住我们对话中的重要信息，并在未来的对话中使用这些记忆。")
    print("输入 'exit' 退出，'profile' 查看我对你的了解\n")
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() == 'exit':
            print("再见！我会记住我们的对话 😊")
            break
        
        if user_input.lower() == 'profile':
            print("\n📋 当前用户画像：")
            print(json.dumps(agent.show_profile(), ensure_ascii=False, indent=2))
            print()
            continue
        
        if not user_input:
            continue
        
        # 获取回复
        response = agent.chat(user_input)
        print(f"\n🤖 助手: {response}\n")


if __name__ == "__main__":
    main()