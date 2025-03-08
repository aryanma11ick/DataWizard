import streamlit as st
import os
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
groq_api_key = ""
deepgram_api_key = ""

groq_client = gt(api_key=groq_api_key)
deepgram_client = DeepgramClient(api_key=deepgram_api_key)

