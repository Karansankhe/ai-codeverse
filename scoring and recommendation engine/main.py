import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st

# Function to generate dummy data
def generate_dummy_data(num_samples=100):
    np.random.seed(42)  # For reproducibility

    data = {
        'Artifact ID': range(1, num_samples + 1),
        'Type': np.random.choice(['file', 'log', 'registry_entry', 'email'], num_samples),
        'Timestamp': pd.date_range(start='2024-10-01', periods=num_samples, freq='H'),
        'File Size': np.random.choice([None, 100, 250, 500, 1000, 1500], num_samples),
        'User ID': np.random.randint(1000, 1100, num_samples),
        'Access Frequency': np.random.randint(1, 20, num_samples),
        'IP Address': np.random.choice(['192.168.1.10', '172.16.0.1', '192.168.1.11', '172.16.0.2'], num_samples),
        'Action Type': np.random.choice(['create', 'modify', 'delete'], num_samples),
        'Known IOC': np.random.choice([0, 1], num_samples),
        'Anomalous Behavior': np.random.choice([0, 1], num_samples),
        'Label': np.random.choice([0, 1], num_samples)  # Randomly assign malicious/benign for training purposes
    }

    return pd.DataFrame(data)

# Function to train the model and make predictions
def train_model(df):
    # Prepare the data for training the model
    features = df[['Known IOC', 'Anomalous Behavior', 'Access Frequency', 'File Size']]
    labels = df['Label']

    # Handle missing values in 'File Size' by filling with the mean
    features['File Size'].fillna(features['File Size'].mean(), inplace=True)

    # Train a simple Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    classification_report_str = classification_report(y_test, predictions)

    # Adding predictions to the original DataFrame
    df['Predicted Malicious'] = model.predict(features)

    # Calculate Risk Score
    df['Risk Score'] = df.apply(calculate_risk_score, axis=1)

    # Apply recommendation engine
    df['Recommendation'] = df.apply(recommend_investigation, axis=1)

    return df[['Artifact ID', 'Type', 'Risk Score', 'Predicted Malicious', 'Recommendation']], classification_report_str

# Calculate Risk Score function
def calculate_risk_score(row):
    action_weight = {'create': 1, 'modify': 2, 'delete': 3}
    user_privilege = 1 if row['User ID'] != 0 else 2  # Simplified assumption of privilege
    score = (
        (5 * row['Known IOC']) +
        (4 * row['Anomalous Behavior']) +
        (2 * (row['Access Frequency'] / 20)) +  # 20 is the max access frequency
        (1 * user_privilege) +
        (3 * action_weight[row['Action Type']])
    )
    return score

# Recommendation Engine function
def recommend_investigation(row):
    # Recommend if the model predicts malicious or risk score exceeds 7
    return row['Predicted Malicious'] == 1 or row['Risk Score'] > 7

# Streamlit UI
st.title("score")
st.write("This tool uses dummy data to simulate cyber triage analysis. Upload your own data or use the generated data below.")

# Sidebar options
st.sidebar.header("Options")
num_samples = st.sidebar.slider("Number of samples", min_value=10, max_value=200, value=100)

# Button to generate and display dummy data
if st.sidebar.button("Generate Data"):
    df = generate_dummy_data(num_samples)
    st.success("Generated")

    # Display generated dummy data
    st.write(" Data:")
    st.dataframe(df)

    # Train the model and display results
    st.write("### Analyzing Data...")
    results, classification_report_str = train_model(df)
    st.write("### Analysis Results:")
    st.dataframe(results)
    
    # Optionally: Download the results
    csv = results.to_csv(index=False)
    st.download_button("Download Results as CSV", csv, "results.csv", "text/csv")

else:
    st.warning("Click 'Generate Dummy Data' to start analyzing.")
