from langchain_core.messages import HumanMessage
from typing import List
from dataclasses import dataclass
from langgraph.graph import StateGraph
from app_pages.graph.state import AgentState
from app_pages.graph.nodes import call_model, call_tools, route_to_tools  # Ensure alignment here
from app_pages.data_models import InputData
import os
import pandas as pd
import cv2
import json
from PIL import Image

class PythonChatbot:
    def __init__(self):
        super().__init__()
        self.reset_chat()
        self.graph = self.create_graph()

    def create_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_mode("agent", call_model) #Model Handling Logic
        workflow.add_node('tools', call_tools) #Tools Handling Node


