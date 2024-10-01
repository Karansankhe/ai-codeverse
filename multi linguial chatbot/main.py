from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from langdetect import detect  # Language detection library

# Load environment variables from .env file
load_dotenv()

# Configure Google API with the provided API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
# Initialize the Google Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to get a response from Google Gemini
def get_gemini_response(content, prompt, language):
    # If the language is not English, include a translation instruction
    if language != 'en':
        prompt += f" Answer in {language}."
    
    response = model.generate_content([content, prompt])
    return response.text

# Function to handle the chat and respond in the user's input language
def handle_chat(user_message):
    # Detect the language of the user's input
    language = detect(user_message)
    
    # Fixed prompt for generating a response
    prompt = f"""
    You are a helpful chatbot providing advice.
    User's Message: {user_message}
    """
    
    # Get response from Gemini
    response = get_gemini_response(user_message, prompt, language)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot")

# Chat input
user_message = st.text_input("Enter your message")

if st.button("Get Response"):
    if user_message:
        response = handle_chat(user_message)
        st.subheader("Chatbot Response")
        st.write(response)
    else:
        st.error("Please enter a message.")
