"""
模块 17：多 Agent 协作
学习如何创建多个专业化 Agent 并让它们协作
"""

import os
from typing import TypedDict, Annotated, Literal, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
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

    # 人工审核节点
    def human_review(state: TeamState) -> dict:
        """人工审核：暂停执行，等待人工审核"""
        print("  [人工审核] 等待人工审核...")

        approval = interrupt(
            f"请审核以下内容：\n\n{state['final_content']}\n\n"
            "输入 'approve' 批准，或输入修改意见："
        )

        print(f"  [人工审核] 审核结果：{approval}")
        return {"messages": [AIMessage(content=f"[人工审核] {approval}")]}

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
    graph.add_node("human_review", human_review)

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
    
    # 各 Agent 完成后回到 supervisor（编辑需要先经过人工审核）
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("editor", "human_review")
    graph.add_edge("human_review", "supervisor")
    graph.add_edge("translator", "supervisor")

    # 编译（需要使用 checkpointer 来支持 interrupt/resume）
    app = graph.compile(checkpointer=MemorySaver())

    # 使用相同的 thread_id 以保持会话状态
    config = {"configurable": {"thread_id": "demo-1"}}

    # 第一次调用：运行到 human_review 节点时暂停
    print("\n--- 第一次调用：运行到人工审核节点 ---")
    result = app.invoke({
        "task": "写一篇关于人工智能发展的简短介绍",
        "messages": []
    }, config)

    print("\n" + "-" * 40)
    print("📝 编辑完成的内容（待审核）：")
    print("-" * 40)
    print(result.get("final_content", "无内容"))

    # 第二次调用：人工审核通过后继续执行
    print("\n--- 人工审核通过，继续执行 ---")
    result = app.invoke(Command(resume="approve"), config)

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
# 示例 2：并行执行模式（Send API）
# ============================================================

def practice_parallel():
    """
    并行执行模式：使用 Send API 让研究员和作家同时工作
    """
    print("\n" + "=" * 60)
    print("示例 2：并行执行模式 - 研究员和作家同时工作")
    print("=" * 60)

    class ParallelState(TypedDict):
        task: str
        messages: Annotated[list, add_messages]
        research_result: str
        draft: str
        final_content: str

    def researcher(state: ParallelState) -> dict:
        """研究员：搜索并整理信息"""
        print("  [研究员] 开始研究任务...")

        search_result = search_web.invoke({"query": state["task"]})

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

    def writer(state: ParallelState) -> dict:
        """作家：基于任务主题撰写初稿（与研究员并行，不依赖研究结果）"""
        print("  [作家] 开始撰写内容...")

        messages = [
            SystemMessage(content="你是一个专业作家，请根据主题撰写一篇结构清晰的短文。用中文写作。"),
            HumanMessage(content=f"主题：{state['task']}")
        ]
        response = model.invoke(messages)

        print(f"  [作家] 完成初稿，共 {len(response.content)} 字")
        return {
            "draft": response.content,
            "messages": [AIMessage(content=f"[作家] {response.content}")]
        }

    def editor(state: ParallelState) -> dict:
        """编辑：等待研究和初稿都完成后，合并优化生成最终文章"""
        print("  [编辑] 合并研究资料和初稿...")

        messages = [
            SystemMessage(content="你是一个资深编辑，请结合研究资料和初稿，优化生成一篇专业、易读的最终文章。用中文回复。"),
            HumanMessage(content=f"主题：{state['task']}\n\n研究资料：{state['research_result']}\n\n初稿：{state['draft']}")
        ]
        response = model.invoke(messages)

        print(f"  [编辑] 编辑完成，最终版本 {len(response.content)} 字")
        return {
            "final_content": response.content,
            "messages": [AIMessage(content=f"[编辑] {response.content}")]
        }

    # 分发节点
    def dispatch_workers(state: ParallelState) -> dict:
        """分发器：准备并行分发"""
        print("  [分发器] 准备并行分发任务...")
        return {}

    # 并行路由函数：返回多个 Send，研究员和作家同时启动
    def continue_to_workers(state: ParallelState) -> list[Send]:
        """将任务并行分发给研究员和作家"""
        print("  [分发器] 同时分发给研究员和作家")
        return [
            Send("researcher", {"task": state["task"]}),
            Send("writer", {"task": state["task"]}),
        ]

    # 构建图
    graph = StateGraph(ParallelState)

    graph.add_node("dispatch", dispatch_workers)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("editor", editor)

    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", continue_to_workers)

    # 研究员和作家都完成后，汇聚到编辑
    graph.add_edge("researcher", "editor")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    app = graph.compile()

    result = app.invoke({
        "task": "写一篇关于人工智能发展的简短介绍",
        "messages": []
    })

    print("\n" + "-" * 40)
    print("📝 最终文章：")
    print("-" * 40)
    print(result["final_content"])

    return result


# ============================================================
# 示例 3：Agent 间通信（辩论模式）
# ============================================================

def practice_communication():
    """
    Agent 间通信模式：两个 Agent 通过共享状态直接对话辩论，
    无需监督者居中协调
    """
    print("\n" + "=" * 60)
    print("示例 3：Agent 间通信 - 辩论模式")
    print("=" * 60)

    MAX_ROUNDS = 2  # 2 轮辩论（正方→反方→正方→反方 各 2 次）

    class DebateState(TypedDict):
        topic: str
        messages: Annotated[list, add_messages]  # 通信通道1：消息列表（自动追加）
        agent_outputs: dict  # 通信通道2：结构化字典（按 Agent 名字存取）
        round_count: int
        verdict: str

    def pro(state: DebateState) -> dict:
        """正方：通过 messages 读取反方上次发言，发表正方观点"""
        round_num = state["round_count"] + 1
        print(f"  [正方] 第 {round_num} 轮发言...")

        # 从共享 messages 中获取对方最后一条消息（Agent 间通信的关键）
        last_msg = ""
        if state["messages"]:
            last_msg = state["messages"][-1].content

        context = f"对方观点：{last_msg}" if last_msg else "请首先发表正方观点"

        messages = [
            SystemMessage(content=f"你是一个辩论正方，主题是：{state['topic']}。请针对对方的观点进行反驳，提出正方论据。用中文简洁回复，200字以内。"),
            HumanMessage(content=context)
        ]
        response = model.invoke(messages)

        pro_key = f"pro_round_{round_num}"
        print(f"  [正方] 发言完成 -> 写入 agent_outputs['{pro_key}']")
        return {
            "messages": [AIMessage(content=f"[正方-第{round_num}轮] {response.content}")],
            "agent_outputs": {**state.get("agent_outputs", {}), pro_key: response.content},
            "round_count": round_num
        }

    def con(state: DebateState) -> dict:
        """反方：通过 messages 读取正方上次发言，发表反方观点"""
        print(f"  [反方] 第 {state['round_count']} 轮发言...")

        # 从共享 messages 中获取正方最后一条消息
        last_msg = state["messages"][-1].content if state["messages"] else ""

        messages = [
            SystemMessage(content=f"你是一个辩论反方，主题是：{state['topic']}。请针对正方的观点进行反驳，提出反方论据。用中文简洁回复，200字以内。"),
            HumanMessage(content=f"对方观点：{last_msg}")
        ]
        response = model.invoke(messages)

        con_key = f"con_round_{state['round_count']}"
        print(f"  [反方] 发言完成 -> 写入 agent_outputs['{con_key}']")
        return {
            "messages": [AIMessage(content=f"[反方-第{state['round_count']}轮] {response.content}")],
            "agent_outputs": {**state.get("agent_outputs", {}), con_key: response.content},
        }

    def judge(state: DebateState) -> dict:
        """裁判：从 agent_outputs 按 key 读取各 Agent 的结构化输出"""
        print("  [裁判] 从 agent_outputs 读取辩论记录...")

        # 从 agent_outputs 字典按 key 读取（Agent 间通信的第二种方式）
        outputs = state.get("agent_outputs", {})
        debate_parts = []
        for key in sorted(outputs.keys()):
            debate_parts.append(f"[{key}]: {outputs[key]}")

        debate_record = "\n\n".join(debate_parts)
        print(f"  [裁判] 从 agent_outputs 读取到 {len(outputs)} 条发言")

        messages = [
            SystemMessage(content="你是一个公正的辩论裁判，请根据以下辩论记录，总结双方观点并给出公正的评判。用中文回复。"),
            HumanMessage(content=f"辩题：{state['topic']}\n\n辩论记录：\n{debate_record}")
        ]
        response = model.invoke(messages)

        print(f"  [裁判] 评判完成")
        return {
            "verdict": response.content,
            "messages": [AIMessage(content=f"[裁判] {response.content}")]
        }

    # 路由函数：正方发言后 → 反方；反方发言后 → 正方（未达上限）或 裁判（达上限）
    def route_after_pro(state: DebateState) -> Literal["con"]:
        return "con"

    def route_after_con(state: DebateState) -> Literal["pro", "judge"]:
        if state["round_count"] < MAX_ROUNDS:
            return "pro"
        else:
            return "judge"

    # 构建图
    graph = StateGraph(DebateState)

    graph.add_node("pro", pro)
    graph.add_node("con", con)
    graph.add_node("judge", judge)

    graph.add_edge(START, "pro")

    graph.add_conditional_edges("pro", route_after_pro, {"con": "con"})
    graph.add_conditional_edges("con", route_after_con, {"pro": "pro", "judge": "judge"})

    graph.add_edge("judge", END)

    app = graph.compile()

    topic = "人工智能是否会取代人类工作"
    result = app.invoke({
        "topic": topic,
        "messages": [],
        "agent_outputs": {},
        "round_count": 0
    })

    print("\n" + "-" * 40)
    print("📋 辩题：", topic)
    print("-" * 40)
    print("agent_outputs 内容（结构化字典通信通道）：")
    for key in sorted(result["agent_outputs"].keys()):
        print(f"  [{key}]: {result['agent_outputs'][key][:80]}...")

    print("\n" + "-" * 40)
    print("⚖️  裁判最终评判：")
    print("-" * 40)
    print(result["verdict"])

    return result


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("多 Agent 协作练习")
    print("=" * 60)

    # 练习 1+3：监督者模式（含翻译 Agent + 人工审核）
    # supervisor_pattern()

    # 练习 2：并行执行模式（研究员 + 作家同时工作）
    # practice_parallel()

    # 练习 4：Agent 间通信（辩论模式）
    practice_communication()

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
