from typing import TypedDict, List, Union
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


load_dotenv()

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

def process(state:AgentState)->AgentState:
    """this node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI:{response.content}")
    print("current_state:",state["messages"])
    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

app = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input!="exit":

    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages":conversation_history})
    conversation_history = result["messages"]
    user_input=input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")

        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")

    file.write("End Of Conversation")

print("Conversation saved to logging.txt")