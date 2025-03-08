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

def get_first_three_sentences(text):
    """Extract the first three sentences from text."""
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:3])

def handle_voice_input_and_output():
    """Handle voice input, transcription, and TTS output."""
    global is_recording, audio_frames
    audio_frames = []
    is_recording = True
    st.write("Recording... Speak now.")

    record_audio("user_voice_input.wav")
    transcription = transcribe_audio("user_voice_input.wav")
    if transcription:
        st.session_state.visualisation_chatbot.user_sent_message(transcription)
        if st.session_state.visualisation_chatbot.chat_history:
            last_message = st.session_state.visualisation_chatbot.chat_history[-1]
            if isinstance(last_message, AIMessage):
                full_response = last_message.content
                tts_text = get_first_three_sentences(full_response)
                text_to_speech(tts_text, "nurse_response.wav")
                if os.path.exists("nurse_response.wav"):
                    st.session_state.play_tts = True
                else:
                    st.error("Failed to generate response audio.")
        st.rerun()

with tab1:
    st.subheader("Your Learning Journey")

    def on_submit_user_query():
        user_query = st.session_state.get('user_input', '')
        if user_query:
            st.session_state.visualisation_chatbot.user_sent_message(user_query)

    chat_container = st.container()

    with chat_container:
        if not st.session_state.visualisation_chatbot.chat_history:
            st.markdown("""
                <div class="chat-container">
                    <div class="teacher-emoji">📚</div>
                    <span class="animated-text">Ready to master Data Structures, Algorithms, and AI/ML? Let’s get started!</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            for msg_index, msg in enumerate(st.session_state.visualisation_chatbot.chat_history):
                if isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(msg.content)
                elif isinstance(msg, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(msg.content)

    # Play TTS audio automatically when triggered
    if st.session_state.get('play_tts', False):
        play_audio("nurse_response.wav")
        st.session_state.play_tts = False

    # Improved UI alignment with spacing and column ratios
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    input_col, mic_col = st.columns([5, 1])
    with input_col:
        st.chat_input(placeholder="Ask me anything about Data Structures, Algorithms, or AI/ML...",
                      on_submit=on_submit_user_query, key='user_input')
    with mic_col:
        if st.button("🎙️", key="mic_button1"):
            handle_voice_input_and_output()

with tab2:
    client = gt(api_key="")  # Replace with your OpenAI API key
    deepgram_client = DeepgramClient(api_key="")

    # Load the main prompt for question generation
    @st.cache_resource
    def initialize_agent():
        """
        Initializes and returns the AI agent.
        """
        return Agent(
            name="Video AI Summarizer",
            model=Groq(id="llama-3.2-90b-vision-preview"),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    def extract_audio_from_video(video_path: str) -> str:
        """
        Extracts audio from a video file and saves it as a WAV file.
        """
        try:
            video = VideoFileClip(video_path)
            audio_path = video_path.replace(".mp4", ".wav")
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            st.error(f"Failed to extract audio: {e}")
            return None
        finally:
            if 'video' in locals():
                video.close()

        def transcribe_audio(audio_path: str, language: str = "en") -> str:
            """
            Transcribes audio from a WAV file to text using Whisper.
            """
            try:
                with open(audio_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        language=language
                    )
                return transcription.text
            except Exception as e:
                st.error(f"Failed to transcribe audio: {e}")
                return ""

        def generate_questions(transcript: str, question_type: str, num_questions: int) -> str:
            """
            Generates questions from the transcript using the LLM.
            """
            try:
                # Define the prompt for question generation
                prompt = f"""
                You are an expert in generating educational content. Based on the following video transcript, create {num_questions} {question_type} questions:

                Transcript:
                {transcript}

                Instructions:
                1. Ensure the questions are clear, concise, and relevant to the content.
                2. For multiple-choice questions, provide 4 options and indicate the correct answer.
                3. For true/false questions, provide the correct answer.
                4. For short answer and essay questions, provide a sample answer or key points.

                Output the questions in the following format:
                - Question 1: [Question text]
                Options (if applicable): [Option A, Option B, Option C, Option D]
                Answer: [Correct answer or key points]
                """

                # Generate questions using the LLM
                response = client.chat.completions.create(
                    model="gemma2-9b-it",  # Replace with your model
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                st.error(f"Failed to generate questions: {e}")
                return ""

        multimodal_agent = initialize_agent()
        # Streamlit app
        st.subheader("Ecllispe")
        st.markdown("Upload a video, and we'll generate a question paper for you!")

        # File uploader
        video_file = st.file_uploader(
            "Upload a video file (MP4, MOV, AVI)",
            type=['mp4', 'mov', 'avi'],
            help="Supported formats: MP4, MOV, AVI."
        )

        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name

            st.video(video_path, format="video/mp4", start_time=0)

            # Language selection
            language = st.radio(
                "Select language for transcription:",
                options=["English", "Hindi"],
                index=0
            )
            language_code = "en" if language == "English" else "hi"

            # Question type selection
            question_type = st.selectbox(
                "Select question type:",
                options=["Multiple Choice", "Short Answer", "True/False", "Essay"],
                help="Choose the type of questions to generate."
            )

            # Number of questions
            num_questions = st.slider(
                "Number of questions to generate:",
                min_value=1,
                max_value=20,
                value=5,
                help="Select the number of questions to generate from the video."
            )

            if st.button("📝 Generate Question Paper", key="generate_questions_button"):
                try:
                    with st.spinner("Processing video and generating questions..."):
                        # Step 1: Extract audio from the video
                        audio_path = extract_audio_from_video(video_path)
                        if not audio_path:
                            raise Exception("Failed to extract audio from the video.")

                        # Step 2: Transcribe audio to text using Whisper
                        transcript = transcribe_audio(audio_path, language=language_code)
                        if not transcript:
                            raise Exception("Failed to transcribe audio to text.")

                        # Step 3: Generate questions using the LLM
                        questions = generate_questions(transcript, question_type, num_questions)
                        if not questions:
                            raise Exception("Failed to generate questions.")

                        # Step 4: Display the generated questions
                        st.subheader("Generated Question Paper")
                        st.markdown(questions, unsafe_allow_html=True)

                except Exception as error:
                    st.error(f"An error occurred during question generation: {error}")
                finally:
                    # Clean up temporary files
                    time.sleep(1)
                    if Path(video_path).exists():
                        try:
                            Path(video_path).unlink(missing_ok=True)
                        except PermissionError:
                            st.warning(f"Could not delete {video_path}. It may still be in use.")
                    if 'audio_path' in locals() and Path(audio_path).exists():
                        try:
                            Path(audio_path).unlink(missing_ok=True)
                        except PermissionError:
                            st.warning(f"Could not delete {audio_path}. It may still be in use.")
        else:
            st.info("Please upload a video file to generate a question paper.")

        # Customize text area height
        st.markdown(
            """
            <style>
            .stTextArea textarea {
                height: 100px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
