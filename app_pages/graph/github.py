import re
import requests
import base64
from markdown import markdown
from bs4 import BeautifulSoup
from groq import Groq
from langchain_core.tools import tool