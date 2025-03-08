from pytube import YouTube
import moviepy
from moviepy.editor import *
import speech_recognition as sr
from transformers import pipeline
import re
from langchain_core.tools import tool
from langchain.tools import tool
import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from groq import Groq as gt

from pytube import YouTube
import moviepy.editor as mp
import speech_recognition as sr
from transformers import pipeline
import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import openai
from moviepy.editor import VideoFileClip
import yt_dlp
import os
import openai
import datetime
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

#Initializing Groq Client
client1 = gt(api_key=groq_api_key)

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL: No video ID found")

