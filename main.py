from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

# 加载 .env
load_dotenv()  

# 配置 DeepSeek API
DEEPSEEK_API_KEY = os.environ.get("API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("呜...API_KEY 没有设置！快去检查 .env 文件！😿")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",  # 或 deepseek-reasoner
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.7,
)

class SimpleAgent:

    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.system_prompt = "你是 chenkx 的 AI 个人助手，请使用简洁且专业的回复风格。"
    
    def set_system_prompt(self, prompt):
        """设置系统提示词"""
        self.system_prompt = prompt
    
    def chat(self, user_input):
        """与 agent 对话"""
        # 构建消息列表
        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(self.conversation_history)
        messages.append(HumanMessage(content=user_input))
        
        # 获取回复
        response = self.llm.invoke(messages)
        
        # 保存对话历史
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=response.content))
        
        return response.content
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空")
    
    def show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("暂无对话历史")
            return
        
        for i, msg in enumerate(self.conversation_history):
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            print(f"\n[{role}]: {msg.content}")


def main():

    print("=" * 50)
    print("DeepSeek 对话 Agent")
    print("=" * 50)
    print("命令:")
    print("  输入消息 - 与 AI 对话")
    print("  'clear' - 清空对话历史")
    print("  'history' - 查看对话历史")
    print("  'quit' - 退出程序")
    print("=" * 50)
    
    # 创建 agent
    agent = SimpleAgent(llm)
    
    while True:
        try:
            user_input = input("\n请输入: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if user_input.lower() == 'clear':
                agent.clear_history()
                continue
            
            if user_input.lower() == 'history':
                agent.show_history()
                continue
            
            # 获取 AI 回复
            response = agent.chat(user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n程序已中断")
            break
        except Exception as e:
            print(f"\n错误: {str(e)}")

if __name__ == "__main__":
    main()