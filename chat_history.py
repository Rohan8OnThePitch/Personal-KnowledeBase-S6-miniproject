from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

def query_or_respond(state: MessagesState, llm_with_tools, retrieve_tool):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate_response(state: MessagesState, llm):
    recent_tool_messages = [
        message for message in reversed(state["messages"])
        if message.type == "tool"
    ][::-1]

    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages)
    system_message_content = (
        f"Use the following context to answer the question:\n{docs_content}"
    )
    
    prompt = [SystemMessage(system_message_content)] + [
        msg for msg in state["messages"] if msg.type in ("human", "system")
    ]
    
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Example usage of StateGraph
memory = MemorySaver()
graph = StateGraph(MessagesState).compile(checkpointer=memory)
