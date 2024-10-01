import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from fpdf import FPDF

# Function to load data from a JSON file
def load_data(file):
    try:
        data = pd.read_json(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to visualize events as a timeline
def plot_timeline(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    ax.plot(data['timestamp'], np.arange(len(data)), marker='o', linestyle='-', color='b')
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data['event'])
    ax.set_xlabel('Time')
    ax.set_title('Interactive Timeline of Events')

    # Annotate events
    for i in range(len(data)):
        ax.annotate(data['ioc'][i], (data['timestamp'][i], i), textcoords="offset points", xytext=(0, 10), ha='center')

    st.pyplot(fig)

# Function to visualize IOCs in a bar chart
def plot_ioc_summary(data):
    plt.figure(figsize=(10, 5))
    ioc_counts = data['ioc'].value_counts()
    sns.barplot(x=ioc_counts.index, y=ioc_counts.values, palette='viridis')
    plt.title('Indicators of Compromise (IOCs)')
    plt.ylabel('Count')
    plt.xlabel('IOC')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Function to visualize severity levels
def plot_severity_distribution(data):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x='severity', palette='pastel')
    plt.title('Distribution of Severity Levels')
    plt.ylabel('Count')
    plt.xlabel('Severity Level')
    st.pyplot(plt)

# Function to generate PDF report
def generate_pdf_report(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Cyber Triage Tool Report", ln=True, align='C')
    pdf.cell(200, 10, txt="Evidence Summary:", ln=True)

    for index, row in data.iterrows():
        pdf.cell(0, 10, f"{row['timestamp']} - {row['event']} (IOC: {row['ioc']}, Severity: {row['severity']})", ln=True)

    return pdf.output(dest='S').encode('latin1')  # Returns PDF as a byte string

# Main Streamlit UI
st.title("Cyber Triage Tool")

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

        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe(include='all'))

        # Interactive timeline
        st.subheader("Event Timeline")
        plot_timeline(data)

        # IOC summary
        st.subheader("Indicators of Compromise (IOCs)")
        plot_ioc_summary(data)

        # Severity distribution
        st.subheader("Severity Distribution")
        plot_severity_distribution(data)

        # Reporting options
        st.subheader("Generate Report")
        report_format = st.selectbox("Choose report format:", ["PDF", "JSON", "CSV"])

        if st.button("Generate Report"):
            report_data = data.to_dict(orient='records')
            if report_format == "PDF":
                pdf_report = generate_pdf_report(data)
                st.download_button("Download PDF Report", pdf_report, file_name="report.pdf", mime="application/pdf")
            elif report_format == "JSON":
                json_report = json.dumps(report_data, indent=4)
                st.download_button("Download JSON Report", json_report, file_name="report.json", mime="application/json")
            elif report_format == "CSV":
                csv_report = data.to_csv(index=False)
                st.download_button("Download CSV Report", csv_report, file_name="report.csv", mime="text/csv")

    else:
        st.warning("No valid data found in the uploaded file.")

else:
    st.info("Upload evidence to get started.")
