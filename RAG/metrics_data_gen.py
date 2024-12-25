import os
import json
import shutil
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from tqdm import tqdm  # Import tqdm for progress bars

# Paths
CHROMA_PATH = "chroma"
DATA_PATH = "data"
OUTPUT_JSON = "qa_pairs.json"

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def extract_text_from_pdfs(pdf_directory):
    """Extract text from all PDF files in the specified directory."""
    documents = []
    for filename in tqdm(os.listdir(pdf_directory), desc="Processing PDFs"):  # Add progress bar
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            reader = PdfReader(pdf_path)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

def split_documents(documents):
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def generate_qa_pairs(text_chunks):
    """Generate question-answer pairs using a local LLM via Ollama."""
    qa_pairs = []
    headers = {'Content-Type': 'application/json'}
    for chunk in tqdm(text_chunks, desc="Generating QA pairs"):  # Add progress bar
        prompt = f"Generate a list of question-answer pairs based on the following text:\n\n{chunk.page_content}"
        data = {
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False
        }
        response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            qa_pairs.extend(response.json().get('response', []))
        else:
            print(f"Failed to generate QA pairs for chunk: {chunk.metadata['source']}")
    return qa_pairs

def save_qa_pairs_to_json(qa_pairs, output_path):
    """Save the generated question-answer pairs to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

def add_qa_pairs_to_chroma(qa_pairs):
    """Add question-answer pairs to the Chroma database."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    for qa in tqdm(qa_pairs, desc="Adding QA pairs to Chroma"):  # Add progress bar
        question = qa['question']
        answer = qa['answer']
        db.add_texts([question, answer])
    db.persist()

def clear_database():
    """Clear the existing Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main():
    # Clear the database if needed
    clear_database()

    # Extract text from PDFs
    documents = extract_text_from_pdfs(DATA_PATH)

    # Split documents into chunks
    text_chunks = split_documents(documents)

    # Generate question-answer pairs
    qa_pairs = generate_qa_pairs(text_chunks)

    # Save QA pairs to JSON
    save_qa_pairs_to_json(qa_pairs, OUTPUT_JSON)

    # Add QA pairs to Chroma database
    add_qa_pairs_to_chroma(qa_pairs)

if __name__ == "__main__":
    main()