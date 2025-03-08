import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, VideoProcessorBase
import threading
import time
import logging
from groq import Groq
from deepgram import DeepgramClient
import queue
import numpy as np
import cv2
import mediapipe as mp
from audio_recorder_streamlit import audio_recorder
from typing import Union, Optional
import os

#Importing API Keys
load_dotenv()
API_KEY1 = os.getenv("API_KEY1")
API_KEY2 = os.getenv("API_KEY2")

#Configuration
AUDIO_SAMPLE_RATE = 44100
CHUNK_DURATION = 0.5
MAX_QUESTIONS = 5

#Initialize API Clients
@st.cache_resource
def get_groq_client():
    return Groq(api_key="API_KEY1")

@st.cache_resource
def get_deepgram_client():
    Deepgram = DeepgramClient(api_key=API_KEY2)
    return Deepgram

groq_client = get_groq_client()
deepgram = get_deepgram_client()

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

