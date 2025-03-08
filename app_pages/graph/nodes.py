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


def route_to_tools(state: AgentState, ) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route back to the agent.
    """

    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


def call_model(state: AgentState):
    current_data_template = """The following data is available:\n{data_summary}\n
 """

    current_data_message = HumanMessage(content=current_data_template.format(
        data_summary=create_data_summary(state)
    ))

    state["messages"] = [current_data_message] + state["messages"]
    llm_outputs = model.invoke(state)

    # Ensure the response includes both questions and answers
    if isinstance(llm_outputs, AIMessage):
        if "Question" in llm_outputs.content and "Answer" not in llm_outputs.content:
            # If only questions are provided, request answers
            follow_up_message = HumanMessage(content="Please provide the answers for these questions.")
            state["messages"].append(follow_up_message)
            llm_outputs = model.invoke(state)

    return {
        "messages": [llm_outputs],
        "intermediate_outputs": [current_data_message.content]
    }
