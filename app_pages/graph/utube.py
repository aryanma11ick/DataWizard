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


# Function to extract metadata
def extract_metadata(url):
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(r.text, features="html.parser")

        title = soup.find("title").text if soup.find("title") else "No Title Found"
        channel = soup.find("link", itemprop="name")['content'] if soup.find("link",
                                                                             itemprop="name") else "No Channel Found"

        return title, channel
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch metadata: {e}")

# Function to download the thumbnail
def download_thumbnail(video_id):
    image_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    try:
        img_data = requests.get(image_url).content
        with open('thumbnail.jpg', 'wb') as handler:
            handler.write(img_data)
    except requests.RequestException as e:
        raise ValueError(f"Failed to download thumbnail: {e}")


# Function to download audio using yt_dlp
def download_audio(youtube_url, output_path=None):
    # Generate a unique filename if none is provided
    if not output_path:
        video_id = extract_video_id(youtube_url)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"youtube_audio_{video_id}_{timestamp}.mp3"

    # Specify the custom path to the directory containing ffmpeg and ffprobe
    ffmpeg_dir = r"C:\ffmpeg-2025-01-30-git-1911a6ec26-full_build\bin"
    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe.exe")

    # Check if ffmpeg and ffprobe exist
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg not found at: {ffmpeg_path}")
    if not os.path.exists(ffprobe_path):
        raise FileNotFoundError(f"ffprobe not found at: {ffprobe_path}")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": output_path,
        "ffmpeg_location": ffmpeg_dir,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Return the correct filename after extraction
        return f"{output_path}.mp3"  # yt-dlp appends .mp3 to the filename
    except yt_dlp.utils.DownloadError as e:
        print(f"Failed to download audio: {e}")
        return None


# Function to transcribe audio using Whisper
def transcribe_audio(audio_path: str, language: str = "en") -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client1.audio.transcriptions.create(
                model="whisper-large-v3",  # OpenAI's Whisper model (supports Large V3)
                file=audio_file,
                language=language
            )
        return transcription.text
    except Exception as e:
        print(f"Failed to transcribe audio: {e}")
        return ""