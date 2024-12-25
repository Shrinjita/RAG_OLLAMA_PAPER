import fitz  # For PDF text extraction
import os
import json
from nltk.tokenize import sent_tokenize
import nltk
from langchain_community.llms.ollama import Ollama  # Assuming this is the library for TinyLlama
from tqdm import tqdm  # For progress bar

# Download required NLTK data
nltk.download('punkt')

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

# Step 2: Contextual text splitting
def split_text_by_context(text, max_chunk_size=512):
    sentences = sent_tokenize(text)
    chunks = []
    temp_chunk = []
    for sentence in sentences:
        temp_chunk.append(sentence)
        if len(" ".join(temp_chunk)) > max_chunk_size:
            chunks.append(" ".join(temp_chunk))
            temp_chunk = []
    if temp_chunk:
        chunks.append(" ".join(temp_chunk))
    return chunks

# Step 3: Extract answers only
def extract_answers_only(context_chunks):
    # Placeholder logic for extracting answers.
    # Replace this with logic that detects specific context and extracts answers.
    answers = []
    for chunk in context_chunks:
        # Extract answers (currently returns the chunk as the answer for demonstration).
        answers.append(chunk.strip())
    return [{"question": "", "answer": answer} for answer in answers]

# Step 4: Generate questions using llama3.2:1b
def generate_questions(qa_pairs):
    # Set up llama3.2:1b with num_threads=8
    model = Ollama(model="llama3.2", num_thread=8)
    for qa_pair in tqdm(qa_pairs, desc="Generating Questions"):
        if qa_pair["question"] == "":
            # Prompt llama3.2 to generate a question
            context = qa_pair["answer"]
            prompt = f"Based on the following context, frame a question: {context}\nOnly provide the question without any additional text or confirmation."
            response = model.invoke(prompt).strip()  # Ensure only the question is returned
            qa_pair["question"] = response
    return qa_pairs

# Step 5: Save QA pairs to a JSON file
def save_to_json(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

# Step 6: Process a single PDF
def process_pdf(pdf_path, output_path):
    # Extract text from the PDF
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    print("Splitting text into contextual chunks...")
    chunks = split_text_by_context(text)
    
    # Extract answers only
    print("Extracting answers...")
    qa_pairs = extract_answers_only(chunks)
    
    # Save intermediate JSON with answers only
    save_to_json(qa_pairs, output_path)
    
    # Generate questions using llama3.2:1b
    print("Generating questions...")
    qa_pairs = generate_questions(qa_pairs)
    
    # Save final JSON with questions and answers
    save_to_json(qa_pairs, output_path)
    print(f"Finished processing {pdf_path}. Output saved to {output_path}.")

# Main Function to process all PDFs in a directory
def main(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each PDF file in the directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(input_dir, pdf_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_qa_pairs.json")
        process_pdf(pdf_path, output_path)

if __name__ == "__main__":
    # Specify your input directory and output directory
    input_dir = "/home/understressengineer/programming/Deep-Learning-Techniques-SRM/RAG/data"  # Directory containing PDFs
    output_dir = "/home/understressengineer/programming/Deep-Learning-Techniques-SRM/RAG/output"  # Directory to save outputs
    
    main(input_dir, output_dir)
