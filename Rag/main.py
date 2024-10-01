import os
from dotenv import load_dotenv
import gradio as gr
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize the summarization pipeline with a T5 model
summarizer = pipeline("summarization", model="t5-small")

# Function to generate a summary from the raw forensic data
def generate_summary(forensic_data):
    if not forensic_data.strip():
        return "Please provide valid forensic log data."
    
    # Create a prompt for summarization
    prompt = f"Summarize the following forensic activities: {forensic_data}"
    
    # Use the summarization pipeline to generate a summary
    summary = summarizer(prompt, max_length=150, min_length=30, do_sample=False)
    
    # Extract the summary from the response
    return summary[0]['summary_text']

# Define Gradio interface
iface = gr.Interface(
    fn=generate_summary,
    inputs=gr.Textbox(label="Forensic Log Data", lines=10, placeholder="Enter forensic data (e.g., system logs, file activity)"),
    outputs="text",
    title="Automated Forensic Data Summarization",
    description="This tool summarizes forensic activities from raw log data using T5."
)

# Launch the Gradio app
iface.launch(share=True)
