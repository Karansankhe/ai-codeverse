import os
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain import hub

# Load environment variables
load_dotenv(find_dotenv())

# Set environment variables for LangChain and Groq
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'advanced-rag'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['GROQ_API_KEY'] = "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"

# Function to handle the summarization process
def summarize_forensic_data(forensic_data):
    ### 1. Text Splitting and Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    documents = text_splitter.split_text(forensic_data)

    ### 2. Embedding and Retrieval
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # Embed the documents and store them in a vector store for retrieval
    vectorstore = FAISS.from_texts(documents=documents, embedding=hf_embeddings)
    retriever = vectorstore.as_retriever()

    ### 3. Language Model for Summarization
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)

    def format_docs(docs):
        return "\n\n".join(docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run the summarization
    summary = rag_chain.invoke("Summarize unusual file access over the past 24 hours.")
    return summary

# Define Gradio interface
iface = gr.Interface(
    fn=summarize_forensic_data,
    inputs=gr.Textbox(label="Forensic Log Data", lines=10, placeholder="Enter forensic data (e.g., system logs, registry entries)"),
    outputs="text",
    title="Automated Forensic Data Summarization",
    description="This tool summarizes unusual file access or suspicious activities from forensic logs using an RAG approach."
)

# Launch the Gradio app
iface.launch()
