from app_pages.graph.github import grok_api_key
import os
 import streamlit as st
 from dotenv import load_dotenv

 load_dotenv()
 grok_api_key=os.getenv("GROK_API_KEY")

t.set_page_config(layout="wide", page_title="Educational Dashboard", page_icon="📚")

data_visualisation_page = st.Page(
    "C:\\Users\\Aryan\\Documents\\Projects\\GDSC_Hackathon\\app_pages\\prompts\\python_visualization_agent.py",
    title="Data Visualisation",
    icon="📈"
)

pg = st.navigation(
    {
        "Visualisation Agent": [data_visualisation_page]
    }
)

pg.run()