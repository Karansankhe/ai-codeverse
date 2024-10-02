import os
import subprocess
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Set Streamlit page configuration
st.set_page_config(page_title="Digital Forensics Data Collector", page_icon="🔍")

class DataCollector:
    def __init__(self, image_path):
        self.image_path = image_path

    def extract_metadata(self):
        # Implement metadata extraction using appropriate forensic tools or libraries.
        # Placeholder for demonstration purposes.
        try:
            # Assuming you have some function that extracts metadata from the image
            metadata = self.analyze_image(self.image_path)
            return self.parse_metadata(metadata)
        except Exception as e:
            return f"Error extracting metadata: {str(e)}"

    def analyze_image(self, image_path):
        # Placeholder for image analysis logic
        # Implement actual logic to extract metadata
        return {
            "example_key": "example_value"  # Replace with actual extracted metadata
        }

    def parse_metadata(self, output):
        metadata = {}
        for key, value in output.items():
            metadata[key] = value.strip()
        return metadata

    def collect_files(self):
        output_dir = "extracted_files"
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Implement file extraction logic using appropriate forensic tools or libraries.
            # Placeholder for demonstration purposes.
            extracted_files = self.extract_files_from_image(self.image_path)
            for file_name in extracted_files:
                with open(os.path.join(output_dir, file_name), 'wb') as f:
                    f.write(b"Sample file data")  # Replace with actual file data
            return output_dir
        except Exception as e:
            return f"Error extracting files: {str(e)}"

    def extract_files_from_image(self, image_path):
        # Placeholder for file extraction logic
        # Implement actual logic to extract files
        return ["file1.txt", "file2.txt"]  # Replace with actual extracted file names

    def collect_data(self):
        metadata = self.extract_metadata()
        files_directory = self.collect_files()
        return metadata, files_directory

def main():
    st.title("Digital Forensics Data Collector")
    st.write("Upload a RAW forensic image to extract metadata and files.")

    uploaded_file = st.file_uploader("Choose a RAW forensic image", type=["raw", "img", "dd"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        image_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Extract Data"):
            collector = DataCollector(image_path)
            metadata, files_directory = collector.collect_data()

            # Display metadata
            st.subheader("Extracted Metadata")
            if isinstance(metadata, dict):
                st.json(metadata)
            else:
                st.error(metadata)
                
            # Display files directory
            st.subheader("Extracted Files")
            if os.path.exists(files_directory):
                st.write(f"Files extracted to: {files_directory}")
                st.download_button("Download Extracted Files", data=open(files_directory, 'rb'), file_name='extracted_files.zip')
            else:
                st.error(files_directory)

            # Generative AI Analysis
            if st.button("Analyze with AI"):
                with st.spinner("Analyzing data..."):
                    try:
                        # Process and prepare data for the AI model
                        analysis_prompt = f"Analyze the following metadata from a forensic image: {metadata}. Provide insights on the extracted files."
                        response = genai.process_text(analysis_prompt)
                        analysis_result = response.get('text', 'No response text found.')
                        st.subheader("AI Analysis Result")
                        st.write(analysis_result)
                    except Exception as e:
                        st.error(f"AI analysis error: {str(e)}")

if __name__ == "__main__":
    main()
