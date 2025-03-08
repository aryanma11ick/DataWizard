import streamlit as st
import os
from dotenv import load_dotenv
import json
from langchain_core.messages import HumanMessage, AIMessage
from app_pages.backend import PythonChatbot, InputData
from app_pages.graph.tools import complete_python_task
import streamlit_ace as st_ace
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import moviepy.editor as mp
import speech_recognition as sr
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, VideoProcessorBase
import threading
import time
import base64
import logging
from groq import Groq as gt
from deepgram import DeepgramClient
import queue
import numpy as np
import cv2
from audio_recorder_streamlit import audio_recorder
from typing import Union, Optional
import os
from pydub import AudioSegment
from groq import Groq as gt
from moviepy.editor import VideoFileClip
from openai import OpenAI
from deepgram import DeepgramClient, SpeakOptions
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import streamlit as st
import pygame
from deepgram import Deepgram
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import re
from functools import lru_cache
import logging
import base64
import wave
import streamlit.components.v1 as components

#Ensuring directory exists
audio_files_dir = "audio_files"
os.makedirs(audio_files_dir, exist_ok=True)

#Initializing API Keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

groq_client = gt(api_key=groq_api_key)
deepgram_client = DeepgramClient(api_key=deepgram_api_key)

#Initialize Pygame for audio playback
pygame.mixer.init()
if 'visualisation_chatbot' not in st.session_state:
    st.session_state.visualisation_chatbot = PythonChatbot()
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

if 'visualisation_chatbot' not in st.session_state:
    st.session_state.visualisation_chatbot = PythonChatbot()
if 'play_tts' not in st.session_state:
    st.session_state.play_tts = False

#Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 20px;
        font-family: 'Arial', sans-serif;
    }
    .header {
        font-size: 28px;
        font-weight: bold;
        color: #fff;
        text-align: center;
        margin-bottom: 20px;
        background-color: #007BFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input, .stChatInput>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .chat-container {
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
        height: 500px;
        overflow-y: auto;
    }
    .stChatMessage.user {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.assistant {
        background-color: #F0F0F0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .welcome-text {
        font-size: 20px;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .teacher-emoji {
        font-size: 70px;
        animation: pulse 2s infinite alternate;
    }
    .animated-text {
        font-size: 26px;
        font-weight: bold;
        color: #0056b3;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes pulse {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

#Header
st.markdown('<div class="header">Clarvis AI Learning Companion</div>', unsafe_allow_html=True)

#Welcome Message
st.markdown("""
    <div class="welcome-text">
        I'm your AI Learning Companion, here to guide, challenge and support you in Mastering Tech Skills
    </div>
""")

#Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Learn with Me", "Ecllipse", "UpSkill", "Debugging Info"])

#Deepgram API setup
deepgram_client = DeepgramClient(api_key=deepgram_api_key)


#Constants
sample_rate= 44100
is_recording = False
audio_frames = []

#Cached Recognizer
@lru_cache(maxsize=None)
def get_recognizer():
    return sr.Recognizer()

def record_audio(file_path, timeout=10, phrase_time_limit=None, retries=3, energy_threshold=2000,
                 pause_threshold=1, phrase_threshold=0.1, dynamic_energy_threshold=True,
                 calibration_duration=1):
    recognizer = get_recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold
    recognizer.phrase_threshold = phrase_threshold
    recognizer.dynamic_energy_threshold = dynamic_energy_threshold

    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                logging.info("Calibrating for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=calibration_duration)
                logging.info("Recording started")
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logging.info("Recording complete")
                with open(file_path, "wb") as f:
                    f.write(audio_data.get_wav_data())
                return
        except sr.WaitTimeoutError:
            logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            logging.error(f"Failed to record audio: {e}")
            if attempt == retries - 1:
                raise
    logging.error("Recording failed after all retries")

def play_audio(file_path):
    #Play audio automatically using HTML5 audio with audioplay
    audio_file = open(file_path, "rb")
    audio_bytes = audio_file.read()
    audio_html = f'<audio autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """Transcribes audio from a WAV file to text using Whisper."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language=language
            )
        logging.info(f"Transcription: {transcription.text}")
        return transcription.text
    except Exception as e:
        st.error(f"Failed to transcribe audio: {e}")
        return ""


def text_to_speech(text, output_file_path):
    #Converts text to speech using Deepgram
    options = {"model": "aura-luna-en", "encoding": "linear16", "container": "wav"}
    try:
        response = deepgram_client.speak.v("1").save(output_file_path, {"text": text}, options)
        logging.info(f"TTS response saved to {output_file_path}")
        return response
    except Exception as e:
        st.error(f"Failed to convert text to speech: {e}")
        return None