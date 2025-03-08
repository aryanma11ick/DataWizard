from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState
import json
from typing import Literal
from .tools import complete_python_task, duckduckgo_search, arxiv_search, wikipedia_search
from .ss import clinical_grade_vital_monitor
from langgraph.prebuilt import ToolInvocation, ToolExecutor
import os

from langchain_ollama import ChatOllama
llm = ChatGroq(model = "deepseek-r1-distill-llama-70b",temperature = 0)

tools = [complete_python_task, duckduckgo_search, arxiv_search, wikipedia_search,clinical_grade_vital_monitor]

model = llm.bind_tools(tools)
tool_executor = ToolExecutor(tools)

with open(os.path.join(os.path.dirname(__file__), "../Prompts/main_prompt.md"), "r") as file:
    prompt = file.read()

chat_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("placeholder", "{messages}"),
])

def create_data_summary(state: AgentState) -> str:
    summary = "Data Structures and Algorithms Summary:\n"
    for d in state["input_data"]:
        # Use existing attributes of InputData class
        summary += f"\n\nSyllabus Name: {d.syllabus_name}\n"
        summary += f"Description: {d.data_description}\n"
        if hasattr(d, 'units') and d.units:
            summary += f"Units: {', '.join(d.units)}\n"
        if hasattr(d, 'objectives') and d.objectives:
            summary += f"Objectives: {', '.join(d.objectives)}\n"
        if hasattr(d, 'text_content') and d.text_content:
            summary += f"Text Content: {d.text_content[:100]}...\n"
    return summary
