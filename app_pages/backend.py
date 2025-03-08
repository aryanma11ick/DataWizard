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

        #Add Nodes based on nodes.py definitions
        workflow.add_conditional_edges('agent', route_to_tools)

        #Set Connections between tools and agent
        workflow.add_edge('tools', 'agent')

        # Define the entry between tools and agent
        workflow.set_entry_point('agent')
        return workflow.compile()

    def user_sent_message(self, user_query):
        #Handles user queries by sending messages through the graph

        starting_image_paths_set = set(sum(self.output_image_paths.values(), []))
        input_state = {
            "message": self.chat_history + [HumanMessage(content=user_query)]
            "output_image_paths": list(starting_image_paths_set),
        }

        #Pass the input state to the compiled graph for processing
        result = self.graph.invoke(input_state, {"recursion_limit": 50})
        self.chat_history = result["messages"]

        #Update the output image paths and intermediate outputs
        new_image_paths = set(result["output_image_paths"]) - starting_image_paths_set
        self.output_image_paths[len(self.chat_history) - 1] = list(new_image_paths)
        if "intermediate_outputs" in result:
            self.intermediate_outputs.extend(result["intermediate_outputs"])

        def reset_chat(self):
            #Resets the chatbot's state, clearing history and intermediate output
            self.chat_history = []
            self.intermediate_outputs = []
            self.output_image_paths = {}
