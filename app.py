import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"
os.environ["OPENAI_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""

import streamlit as st
st.set_page_config(layout="wide", page_title="Education Dashboard", page_icon=" ")

data_visualization_page = st.Page(
    
)