import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st
from transformers import BertTokenizer, BertModel
import torch

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
    features = df[['Known IOC', 'Anomalous Behavior', 'Access Frequency', 'File Size']]
    labels = df['Label']
    features['File Size'].fillna(features['File Size'].mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, predictions))

    df['Predicted Malicious'] = model.predict(features)
    df['Risk Score'] = df.apply(calculate_risk_score, axis=1)
    df['Recommendation'] = df.apply(recommend_investigation, axis=1)

    return df[['Artifact ID', 'Type', 'Risk Score', 'Predicted Malicious', 'Recommendation']]

# Calculate Risk Score function
def calculate_risk_score(row):
    action_weight = {'create': 1, 'modify': 2, 'delete': 3}
    user_privilege = 1 if row['User ID'] != 0 else 2
    score = (
        (5 * row['Known IOC']) +
        (4 * row['Anomalous Behavior']) +
        (2 * (row['Access Frequency'] / 20)) +
        (1 * user_privilege) +
        (3 * action_weight[row['Action Type']])
    )
    return score

# Recommendation Engine function
def recommend_investigation(row):
    return row['Predicted Malicious'] == 1 or row['Risk Score'] > 7

# Function to load BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

# Function to encode text using BERT
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Contextual search function
def contextual_search(query, df, tokenizer, model):
    query_embedding = encode_text(query, tokenizer, model).detach().numpy()
    df['Embedding'] = df['Type'].apply(lambda x: encode_text(x, tokenizer, model).detach().numpy())
    
    # Compute cosine similarity
    df['Similarity'] = df['Embedding'].apply(lambda x: torch.nn.functional.cosine_similarity(
        torch.tensor(query_embedding), torch.tensor(x)).item()
    )
    
    return df.sort_values('Similarity', ascending=False).head(5)

# Streamlit UI
st.title("Cyber Triage Tool")
st.write("This tool uses dummy data to simulate cyber triage analysis.")

# Generate and display dummy data
df = generate_dummy_data()
st.write("Generated Dummy Data:")
st.dataframe(df)

# Train the model and display results
results = train_model(df)
st.write("Analysis Results:")
st.dataframe(results)

# Load BERT model
tokenizer, model = load_bert_model()

# Input for contextual search
st.subheader("Contextual AI-based Search")
query = st.text_input("Enter your search query:")
if query:
    search_results = contextual_search(query, df, tokenizer, model)
    st.write("Search Results:")
    st.dataframe(search_results[['Artifact ID', 'Type', 'Similarity']])

# Optionally: Download the results
csv = results.to_csv(index=False)
st.download_button("Download Results", csv, "results.csv", "text/csv")
