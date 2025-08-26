from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[HumanMessage]
    
llm = OllamaLLM(model="gpt-oss:latest")

def process(state: AgentState) -> AgentState:
    """This node call a llm an give response to user."""
    response = llm.invoke(state["messages"])
    print(f'\nAI: {response}')
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")