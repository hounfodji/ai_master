from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = OllamaLLM(model="gpt-oss:latest")

def process(state: AgentState) -> AgentState:
    """This node call a llm an give response to user."""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response))
    print(f'\nAI: {response}')
    print(f'CURRENT STATE: {state["messages"]}')
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter your message: ")
    
    
with open("conversation_history.txt", "w") as f:
    f.write("Your conversation Log: \n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n\n")
    f.write("-------------------------------------------------------------------\n")
            
print("Conversation history saved to conversation_history.txt")