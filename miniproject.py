from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

@tool
def update(content:str)->str:
    """updates doc with appropriate content"""
    document_content = content
    return f"The content is : \n{document_content}"

@tool
def save(filename:str)->str:
    """saves current doc to text file and finish the process
    Args:
    filename:name for the text file"""
    
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"


    try:
        with open(filename, 'w') as file:
            file.write(document_content)

        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update,save]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

- If the user wants to update or modify content, use the 'update' tool with the complete updated content.
- If the user wants to save and finish, you need to use the 'save' tool.
- Make sure to always show the current document state after modifications.

The current document content is:
{document_content}
"""
    )

    