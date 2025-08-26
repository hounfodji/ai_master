from email import message
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    
@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds two numbers together"""
    
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This is a subtraction function that subtracts two numbers together"""
    
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """This is a multiplication function that multiplies two numbers together"""
    
    return a * b

tools = [add, subtract, multiply]

model = ChatOllama(model="gpt-oss:latest").bind_tools(tools)

# Create a ToolNode to handle tool execution
tool_node = ToolNode(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability"                            
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 10. Also tell me a joke.")]}
print_stream(app.stream(inputs, stream_mode="values"))