"""
模块 17：多 Agent 协作
学习如何创建多个专业化 Agent 并让它们协作
"""

import os
from typing import TypedDict, Annotated, Literal, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# # 加载环境变量
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#
# if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
#     raise ValueError(
#         "\n请先在 .env 文件中设置有效的 GROQ_API_KEY\n"
#         "访问 https://console.groq.com/keys 获取免费密钥"
#     )
#
# # 初始化模型
# model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
from model_init import model
# ============================================================
# 定义工具
# ============================================================

@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    # 模拟搜索结果
    mock_results = {
        "人工智能": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。主要领域包括机器学习、深度学习、自然语言处理等。",
        "机器学习": "机器学习是AI的子领域，通过算法让计算机从数据中学习。常见方法包括监督学习、无监督学习和强化学习。",
        "default": f"找到关于'{query}'的相关信息：这是一个重要的技术领域，正在快速发展中。"
    }
    for key in mock_results:
        if key in query:
            return mock_results[key]
    return mock_results["default"]

@tool
def check_grammar(text: str) -> str:
    """检查文本的语法和表达"""
    # 模拟语法检查
    return f"语法检查完成。文本长度：{len(text)}字符。建议：表达清晰，结构合理。"

# ============================================================
# 示例 1：监督者模式
# ============================================================

def supervisor_pattern():
    """
    监督者模式：由一个 Supervisor 协调多个专业 Agent
    """
    print("\n" + "=" * 60)
    print("示例 1：监督者模式 - 内容创作团队")
    print("=" * 60)

    # 定义状态
    class TeamState(TypedDict):
        task: str
        messages: Annotated[list, add_messages]
        research_result: str
        draft: str
        final_content: str
        translation_content: str
        next_agent: str

    # 初始化模型
    
    # 监督者节点
    def supervisor(state: TeamState) -> dict:
        """监督者：决定下一步由哪个 Agent 处理"""
        print("  [监督者] 分析任务状态...")
        
        # 决策逻辑
        if not state.get("research_result"):
            next_agent = "researcher"
            print("  [监督者] 决定：需要先研究 -> 分配给研究员")
        elif not state.get("draft"):
            next_agent = "writer"
            print("  [监督者] 决定：有研究结果，需要写作 -> 分配给作家")
        elif not state.get("final_content"):
            next_agent = "editor"
            print("  [监督者] 决定：有初稿，需要编辑 -> 分配给编辑")
        elif not state.get("translation_content"):
            print("  [监督者] 决定：有终稿，需要翻译 -> 分配给翻译员")
            next_agent = "translator"
        else:
            next_agent = "complete"
            print("  [监督者] 决定：任务完成")
        
        return {"next_agent": next_agent}

    # 研究员 Agent
    def researcher(state: TeamState) -> dict:
        """研究员：收集和整理信息"""
        print("  [研究员] 开始研究任务...")
        
        # 使用搜索工具
        search_result = search_web.invoke({"query": state["task"]})
        
        # 使用 LLM 整理信息
        messages = [
            SystemMessage(content="你是一个研究员，请根据搜索结果整理出关键信息要点。用中文回复。"),
            HumanMessage(content=f"任务：{state['task']}\n\n搜索结果：{search_result}")
        ]
        response = model.invoke(messages)
        
        print(f"  [研究员] 研究完成，整理了 {len(response.content)} 字的资料")
        
        return {
            "research_result": response.content,
            "messages": [AIMessage(content=f"[研究员] {response.content}")]
        }

    # 作家 Agent
    def writer(state: TeamState) -> dict:
        """作家：根据研究结果撰写内容"""
        print("  [作家] 开始撰写内容...")
        
        messages = [
            SystemMessage(content="你是一个专业作家，请根据研究资料撰写一篇结构清晰的短文。用中文写作。"),
            HumanMessage(content=f"主题：{state['task']}\n\n研究资料：{state['research_result']}")
        ]
        response = model.invoke(messages)
        
        print(f"  [作家] 完成初稿，共 {len(response.content)} 字")
        
        return {
            "draft": response.content,
            "messages": [AIMessage(content=f"[作家] {response.content}")]
        }

    # 编辑 Agent
    def editor(state: TeamState) -> dict:
        """编辑：审核和优化内容"""
        print("  [编辑] 开始审核编辑...")
        
        # 语法检查
        grammar_check = check_grammar.invoke({"text": state["draft"]})
        
        messages = [
            SystemMessage(content="你是一个资深编辑，请审核并优化以下文章，使其更加专业和易读。用中文回复。"),
            HumanMessage(content=f"初稿：{state['draft']}\n\n语法检查：{grammar_check}")
        ]
        response = model.invoke(messages)
        
        print(f"  [编辑] 编辑完成，最终版本 {len(response.content)} 字")
        
        return {
            "final_content": response.content,
            "messages": [AIMessage(content=f"[编辑] {response.content}")]
        }

    # 翻译员 Agent
    def translator(state: TeamState) -> dict:
        """翻译员：将内容翻译成英文"""
        print("  [翻译员] 开始翻译文本任务...")


        # 使用 LLM 整理信息
        messages = [
            SystemMessage(content="你是一个翻译员，请将用户提供的内容翻译成英文。"),
            HumanMessage(content=f"任务：{state['final_content']}")
        ]
        response = model.invoke(messages)

        print(f"  [翻译员] 翻译完成，翻译结果 {len(response.content)} 字")

        return {
            "final_content": state['final_content'],
            "translation_content": response.content,
            "messages": [AIMessage(content=f"[翻译员] {response.content}")]
        }

    # 路由函数
    def route_to_agent(state: TeamState) -> Literal["researcher", "writer", "editor", "complete","translator"]:
        return state["next_agent"]

    # 构建图
    graph = StateGraph(TeamState)
    
    # 添加节点
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("editor", editor)
    graph.add_node("translator", translator)

    # 从 START 到 supervisor
    graph.add_edge(START, "supervisor")
    
    # supervisor 根据条件路由
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "editor": "editor",
            "translator": "translator",
            "complete": END
        }
    )
    
    # 各 Agent 完成后回到 supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("editor", "supervisor")
    graph.add_edge("translator", "supervisor")

    # 编译并运行
    app = graph.compile()
    
    result = app.invoke({
        "task": "写一篇关于人工智能发展的简短介绍",
        "messages": []
    })
    
    print("\n" + "-" * 40)
    print("📝 最终内容：")
    print("-" * 40)
    print(result["final_content"])
    print("\n" + "-" * 40)
    print("📝 翻译内容：")
    print("-" * 40)
    print(result["translation_content"])
    return result



# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("多 Agent 协作教程")
    print("=" * 60)
    
    supervisor_pattern()
    # collaborative_chain()
    # dynamic_dispatch()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
