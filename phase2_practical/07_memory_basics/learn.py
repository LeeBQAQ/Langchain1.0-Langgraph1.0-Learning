from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from model_init import model

def getAgent(model,
             tools,
             system_prompt,
             memory):
    if memory is not None:
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=memory
        )
    else:
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt
        )
    return agent

def chat_without_memory(model):
    agent = getAgent(model,
                     [],
                     "你是一个有帮助的助手。")

    print("\n第一轮对话：")
    response1 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫张三"}]
    })
    # 最后一条消息，即为最终输出
    print(f"\nAgent_output: {response1}")
    print(f"\nAgent_final_output: {response1['messages'][-1].content}")
    print("\n第二轮对话：")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫什么"}]
    })
    print(f"\nAgent_output: {response2}")
    print(f"\nAgent_final_output: {response2['messages'][-1]}.content")

def chat_with_memory(model):
    agent = getAgent(model,
                     [],
                     "你是一个有帮助的助手。",
                     InMemorySaver())
    # 会话配置
    config = {"configurable": {"thread_id": "conversation_1"}}
    print("\n第一轮对话：")
    message1 = {"messages": [{"role": "user", "content": "我叫张三"}]}
    response1 = agent.invoke(message1, config=config)
    print(f"Agent: {response1['messages'][-1].content}")

    print("\n第二轮对话（同一个 thread_id）：")
    message2 = {"messages": [{"role": "user", "content": "我叫什么"}]}
    response2 = agent.invoke(message2, config=config)
    print(f"Agent: {response2['messages'][-1].content}")

if __name__ =="__main__":
    print("chat_without_memory")
    # chat_without_memory(model)
    chat_with_memory(model)