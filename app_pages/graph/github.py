import os
import re
import requests
import base64
from markdown import markdown
from bs4 import BeautifulSoup
from groq import Groq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
grok_api_key = os.getenv('GROK_API_KEY')

client1 = Groq(api_key=grok_api_key)

KEY_DIRECTORIES = ['src', 'tests', 'docs', 'examples']

# Function to convert markdown to plain text
def md_to_text(md: str) -> str:
    """Convert markdown text to plain text for summarization."""
    try:
        html = markdown(md)
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Failed to convert markdown to text: {e}")
        return md  # Return original text as fallback
