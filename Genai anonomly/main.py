import os
from dotenv import load_dotenv
import gradio as gr
import pandas as pd
from transformers import pipeline
from sklearn.ensemble import IsolationForest

# Load environment variables
load_dotenv()

# Initialize the summarization pipeline with a T5 model
summarizer = pipeline("summarization", model="t5-small")

# Sample data for anomaly detection (you may replace this with actual data)
sample_data = pd.DataFrame({
    'user_id': ['user1', 'user2', 'user1', 'user3', 'user2'],
    'action_type': ['login', 'file_access', 'file_creation', 'login', 'file_deletion'],
    'timestamp': pd.to_datetime(['2024-10-01 10:00:00', '2024-10-01 10:05:00', 
                                  '2024-10-01 10:10:00', '2024-10-01 10:15:00', 
                                  '2024-10-01 10:20:00']),
    'file_size': [200, 500, 300, 700, 100],  # in KB
    'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.1', '192.168.1.3', '192.168.1.2'],
    'network_traffic': [1000, 1500, 800, 1200, 600]  # in MB
})

# Anomaly Detection Model
anomaly_detector = IsolationForest(contamination=0.2)

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

# Function to detect anomalies in the dataset
def detect_anomalies(data):
    # Extract features for anomaly detection
    features = data[['file_size', 'network_traffic']]
    
    # Fit the model
    anomaly_detector.fit(features)
    
    # Predict anomalies
    data['anomaly'] = anomaly_detector.predict(features)
    
    # Get the anomalies
    anomalies = data[data['anomaly'] == -1]
    return anomalies

# Main function to handle summarization and anomaly detection
def process_forensic_data(forensic_data):
    if not forensic_data.strip():
        return "Please provide valid forensic log data."
    
    # Convert input to DataFrame (in a real application, you would parse this)
    # Here we assume forensic_data is a string representation of the DataFrame
    data = sample_data  # Replace this with actual parsing of forensic_data
    
    # Generate summary
    summary = generate_summary(forensic_data)
    
    # Detect anomalies
    anomalies = detect_anomalies(data)
    
    # Prepare the output
    anomaly_report = anomalies.to_string(index=False) if not anomalies.empty else "No anomalies detected."
    
    return f"Summary:\n{summary}\n\nAnomaly Report:\n{anomaly_report}"

# Define Gradio interface
iface = gr.Interface(
    fn=process_forensic_data,
    inputs=gr.Textbox(label="Forensic Log Data", lines=10, placeholder="Enter forensic data (e.g., system logs, file activity)"),
    outputs="text",
    title="Automated Forensic Data Summarization and Anomaly Detection",
    description="This tool summarizes forensic activities from raw log data and detects anomalies."
)

# Launch the Gradio app
iface.launch(share=True)
