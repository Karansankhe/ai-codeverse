import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from fpdf import FPDF

# Function to load data from a JSON file
def load_data(file):
    try:
        data = pd.read_json(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to preprocess the data
def preprocess_data(data):
    # Encode categorical features
    label_encoders = {}
    for column in ['event', 'ioc']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

# Build and train autoencoder model for anomaly detection
def build_autoencoder(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(input_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Detect anomalies based on reconstruction error
def detect_anomalies_with_autoencoder(data, threshold, epochs=20):
    X = data[['event', 'ioc']].values
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

    # Build the autoencoder model
    autoencoder = build_autoencoder(X_train.shape[1])
    
    # Train the model
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)

    # Get reconstruction errors for the entire dataset
    reconstructions = autoencoder.predict(X)
    reconstruction_errors = np.mean(np.abs(reconstructions - X), axis=1)
    
    # Add reconstruction errors and anomaly flag to the data
    data['reconstruction_error'] = reconstruction_errors
    data['is_anomaly'] = data['reconstruction_error'] > threshold

    return data, history

# Function to generate PDF report
def generate_pdf_report(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Cyber Triage Tool Report", ln=True, align='C')
    pdf.cell(200, 10, txt="Anomaly Detection Summary:", ln=True)

    for index, row in data.iterrows():
        if row['is_anomaly']:
            pdf.cell(0, 10, f"{row['timestamp']} - {row['event']} (IOC: {row['ioc']}, Error: {row['reconstruction_error']:.4f})", ln=True)

    return pdf.output(dest='S').encode('latin1')

# Function to visualize anomalies
def plot_anomalies(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='timestamp', y='reconstruction_error', hue='is_anomaly', palette='coolwarm', s=100)
    plt.title('Anomaly Detection Results')
    plt.ylabel('Reconstruction Error')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Main Streamlit UI
st.title("Cyber Triage Tool with Deep Learning Anomaly Detection")

# Sidebar for options
st.sidebar.header("Options")
upload_file = st.sidebar.file_uploader("Upload Evidence (JSON)", type=["json"])

if upload_file is not None:
    data = load_data(upload_file)
    if data is not None and not data.empty:
        st.success("Evidence uploaded successfully!")

        # Display the data in a table
        st.subheader("Evidence Summary")
        st.write(data)

        # Preprocess the data
        processed_data, label_encoders = preprocess_data(data)

        # Parameters for Autoencoder
        st.sidebar.subheader("Anomaly Detection Parameters")
        epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=20, step=5)
        threshold = st.sidebar.slider("Anomaly Detection Threshold", min_value=0.01, max_value=0.2, value=0.05)

        # Anomaly Detection
        processed_data, history = detect_anomalies_with_autoencoder(processed_data, threshold, epochs)

        # Show anomalies
        st.subheader("Detected Anomalies")
        anomalies = processed_data[processed_data['is_anomaly']]
        st.write(anomalies)

        # Plot anomalies
        st.subheader("Anomaly Detection Visualization")
        plot_anomalies(processed_data)

        # Reporting options
        st.subheader("Generate Report")
        report_format = st.selectbox("Choose report format:", ["PDF", "JSON", "CSV"])

        if st.button("Generate Report"):
            report_data = anomalies.to_dict(orient='records')
            if report_format == "PDF":
                pdf_report = generate_pdf_report(processed_data)
                st.download_button("Download PDF Report", pdf_report, file_name="report.pdf", mime="application/pdf")
            elif report_format == "JSON":
                json_report = json.dumps(report_data, indent=4)
                st.download_button("Download JSON Report", json_report, file_name="report.json", mime="application/json")
            elif report_format == "CSV":
                csv_report = anomalies.to_csv(index=False)
                st.download_button("Download CSV Report", csv_report, file_name="report.csv", mime="text/csv")

    else:
        st.warning("No valid data found in the uploaded file.")
else:
    st.info("Upload evidence to get started.")
