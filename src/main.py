"""
AI Agent 主入口
"""

import json
from src.agents import SimpleAgent, MemoryAgent

def main():
    """
    主函数
    """
    print("=" * 50)
    print("DeepSeek AI Agent")
    print("=" * 50)
    print("Agent 类型:")
    print("  1. SimpleAgent - 基本对话功能")
    print("  2. MemoryAgent - 具有长期记忆功能")
    print("=" * 50)
    
    # 选择Agent类型
    agent_type = input("请选择Agent类型 (1/2): ").strip()
    while agent_type not in ["1", "2"]:
        agent_type = input("无效选择，请重新输入 (1/2): ").strip()
    
    # 创建Agent
    if agent_type == "1":
        agent = SimpleAgent()
        print("\n已选择 SimpleAgent")
    else:
        agent = MemoryAgent()
        print("\n已选择 MemoryAgent")
    
    print("\n命令:")
    print("  输入消息 - 与 AI 对话")
    print("  'clear' - 清空对话历史")
    print("  'history' - 查看对话历史")
    print("  'profile' - 查看用户画像 (仅MemoryAgent)")
    print("  'quit' - 退出程序")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n请输入: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if user_input.lower() == 'clear':
                agent.clear_memory()
                continue
            
            if user_input.lower() == 'history':
                agent.show_history()
                continue
            
            if user_input.lower() == 'profile':
                if isinstance(agent, MemoryAgent):
                    profile = agent.get_profile()
                    print("\n📋 当前用户画像：")
                    print(json.dumps(profile, ensure_ascii=False, indent=2))
                else:
                    print("\n⚠️  SimpleAgent 不支持用户画像功能")
                continue
            
            # 获取AI回复
            response = agent.chat(user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n程序已中断")
            break
        except Exception as e:
            print(f"\n错误: {str(e)}")

if __name__ == "__main__":
    main()