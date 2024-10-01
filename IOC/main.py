import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

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

# Function to identify IOCs based on rules
def identify_iocs(data):
    suspicious_iocs = ["malware.exe", "suspicious_file.txt", "192.168.1.1"]
    data['is_ioc'] = data['ioc'].isin(suspicious_iocs)
    return data

# Function to detect anomalies using Isolation Forest
def detect_anomalies(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    data['anomaly_score'] = model.fit_predict(data[['event', 'ioc']])
    # Anomalies are labeled as -1
    data['is_anomaly'] = data['anomaly_score'] == -1
    return data

# Function to apply ML model for pattern recognition
def pattern_recognition(data):
    X = data[['event', 'ioc']]
    y = data['is_anomaly']

    # Apply a basic MLPClassifier for pattern recognition
    model = MLPClassifier(random_state=42, max_iter=300)
    model.fit(X, y)

    # Predict the anomaly score
    data['ml_anomaly_score'] = model.predict(X)
    return data

# Function to generate PDF report
def generate_pdf_report(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Cyber Triage Tool Report", ln=True, align='C')
    pdf.cell(200, 10, txt="Anomaly Detection and IOC Summary:", ln=True)

    for index, row in data.iterrows():
        if row['is_anomaly'] or row['is_ioc']:
            pdf.cell(0, 10, f"{row['timestamp']} - {row['event']} (IOC: {row['ioc']}, Anomaly Score: {row['anomaly_score']})", ln=True)

    return pdf.output(dest='S').encode('latin1')

# Function to visualize anomalies using PCA
def plot_anomalies(data):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data[['event', 'ioc']])

    fig = px.scatter(
        x=data_pca[:, 0], y=data_pca[:, 1], 
        color=data['is_anomaly'].astype(str),
        labels={'color': 'Anomaly'},
        title='Anomaly Detection (PCA-based Visualization)'
    )
    st.plotly_chart(fig)

# Main Streamlit UI
st.title("Cyber Triage Tool with Anomaly Detection and IOC Identification")

# Sidebar for options
st.sidebar.header("Options")
upload_file = st.sidebar.file_uploader("Upload Evidence (JSON)", type=["json", "csv"])

if upload_file is not None:
    data = load_data(upload_file)
    if data is not None and not data.empty:
        st.success("Evidence uploaded successfully!")

        # Display the data in a table
        st.subheader("Evidence Summary")
        st.write(data)

        # Preprocess the data
        processed_data, label_encoders = preprocess_data(data)

        # Identify Indicators of Compromise (IOCs)
        processed_data = identify_iocs(processed_data)

        # Anomaly Detection
        processed_data = detect_anomalies(processed_data)

        # Apply ML-based pattern recognition
        processed_data = pattern_recognition(processed_data)

        # Show detected IOCs and anomalies
        st.subheader("Detected IOCs and Anomalies")
        anomalies = processed_data[(processed_data['is_anomaly']) | (processed_data['is_ioc'])]
        st.write(anomalies)

        # Plot anomalies
        st.subheader("Anomaly Detection Visualization (PCA)")
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
