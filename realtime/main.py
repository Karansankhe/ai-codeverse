import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import random
import matplotlib.pyplot as plt

# Function to generate dummy data
def generate_dummy_data(num_samples=100):
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'Source IP': [f'192.168.1.{random.randint(1, 255)}' for _ in range(num_samples)],
        'Destination IP': [f'192.168.1.{random.randint(1, 255)}' for _ in range(num_samples)],
        'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples),
        'Port': np.random.randint(1, 65536, num_samples),
        'File Type': np.random.choice(['exe', 'jpg', 'pdf', 'docx', 'mp4'], num_samples),
        'Event Type': np.random.choice(['access', 'modification', 'creation', 'deletion'], num_samples),
        'Label': np.random.choice(['benign', 'malicious'], num_samples)  # Random labels for training
    }

    return pd.DataFrame(data)

# Function to train the model
def train_model(df):
    # Encoding categorical features
    le_protocol = LabelEncoder()
    le_file_type = LabelEncoder()
    le_event_type = LabelEncoder()
    le_label = LabelEncoder()

    # Convert categorical features to numerical
    df['Protocol'] = le_protocol.fit_transform(df['Protocol'])
    df['File Type'] = le_file_type.fit_transform(df['File Type'])
    df['Event Type'] = le_event_type.fit_transform(df['Event Type'])
    df['Label'] = le_label.fit_transform(df['Label'])

    # Prepare feature set excluding IPs and Label
    features = df.drop(columns=['Label', 'Source IP', 'Destination IP'])
    labels = df['Label']

    # Train Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(features, labels)

    return model, le_protocol, le_file_type, le_event_type, le_label

# Function to classify incoming data
def classify_data(model, le_protocol, le_file_type, le_event_type, new_data):
    new_data['Protocol'] = le_protocol.transform(new_data['Protocol'])
    new_data['File Type'] = le_file_type.transform(new_data['File Type'])
    new_data['Event Type'] = le_event_type.transform(new_data['Event Type'])
    
    # Drop IP addresses and keep only numeric features for prediction
    new_data = new_data.drop(columns=['Source IP', 'Destination IP'])
    
    predictions = model.predict(new_data)
    return predictions

# Streamlit UI
st.title("Real-Time Data Classification for Cybersecurity")
st.write("This tool classifies network traffic, file types, and system events.")

# Generate dummy data for training
df = generate_dummy_data(500)

# Visualize training data distribution
st.subheader("Training Data Distribution")
label_counts = df['Label'].value_counts()
fig, ax = plt.subplots()
label_counts.plot(kind='bar', ax=ax)
ax.set_title("Label Distribution in Training Data")
ax.set_xlabel("Label")
ax.set_ylabel("Count")
st.pyplot(fig)

# Train the model
model, le_protocol, le_file_type, le_event_type, le_label = train_model(df)

# Input for real-time classification
st.subheader("Real-Time Classification Input")
source_ip = st.text_input("Source IP (e.g., 192.168.1.10)", "192.168.1.10")
destination_ip = st.text_input("Destination IP (e.g., 192.168.1.20)", "192.168.1.20")
protocol = st.selectbox("Protocol", ['TCP', 'UDP', 'ICMP'])
port = st.number_input("Port", min_value=1, max_value=65535, value=80)
file_type = st.selectbox("File Type", ['exe', 'jpg', 'pdf', 'docx', 'mp4'])
event_type = st.selectbox("Event Type", ['access', 'modification', 'creation', 'deletion'])

if st.button("Classify"):
    new_data = pd.DataFrame({
        'Source IP': [source_ip],
        'Destination IP': [destination_ip],
        'Protocol': [protocol],
        'Port': [port],
        'File Type': [file_type],
        'Event Type': [event_type]
    })
    
    predictions = classify_data(model, le_protocol, le_file_type, le_event_type, new_data)
    result_label = le_label.inverse_transform(predictions)[0]
    st.write("Classification Result:")
    st.write("Malicious" if result_label == 'malicious' else "Benign")

    # Visualization of classification result
    fig2, ax2 = plt.subplots()
    ax2.pie([1, 1], labels=['Benign', 'Malicious'], autopct='%1.1f%%', startangle=90, 
             colors=['lightblue', 'lightcoral'] if result_label == 'malicious' else ['lightgreen', 'lightgray'])
    ax2.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    st.pyplot(fig2)

