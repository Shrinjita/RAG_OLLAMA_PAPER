# 🚀 RAG: Local Server Deployable Retrieval-Augmented Generation (RAG)

## 📌 Overview
This project provides a **Retrieval-Augmented Generation (RAG)** pipeline designed for **local deployment** with **data security**. It can infer over any document(s) provided by the user. The system is **general-purpose**, allowing users to upload PDFs and run **semantic search & question-answering** without external dependencies.

---

## 🔥 Features
- **Local Server Deployment** (No Cloud Dependency)
- **Secure Data Processing**
- **Plug & Play with Any Document**
- **Supports Fine-Tuning and Custom Datasets**
- **Evaluation Metrics:** BLEU, ROUGE (1,2,L), Cosine Similarity

---

## 🏗️ Project Structure
```plaintext
├── chroma/                         # ChromaDB storage (delete if errors occur)
├── data/                            # User-uploaded PDFs (add files here)
├── evaluation_outputs/              # Precomputed evaluation results
│   ├── Cyber_Security/
│   ├── Laws/
│   ├── Monopoly/
├── output/                          # Generated outputs (QA pairs, embeddings)
├── scripts/
│   ├── populate_database.py         # Prepares database from PDFs
│   ├── query_data.py                # Query your documents
│   ├── metrics_data_gen.py          # Evaluation metrics computation
│   ├── Question_Answer_Generator.py # Generates QA pairs
│   ├── get_embedding_function.py    # Embedding function setup
├── README.md                        # This guide
├── requirements.txt                  # Dependencies
```

---

## 🛠️ Installation

### 1️⃣ Prerequisites
Ensure your system has:
- **Python 3.10+**
- **[Ollama](https://ollama.com)**
- **ChromaDB**
- **Llama 3.2**
- **nomic-embed-text**

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 🏗️ Step 1: Add Documents
Place your PDFs inside the `data/` directory.

### 🔄 Step 2: Populate the Database
Run the script to process and embed your documents:
```bash
python scripts/populate_database.py
```
> If you face issues, **delete the `chroma/` folder** and rerun the script.

### 🤖 Step 3: Query the RAG System
Once the database is populated, you can retrieve information using:
```bash
python scripts/query_data.py "Your query here"
```

### 📊 Step 4: Evaluate Performance (Optional)
If you want to compute metrics like BLEU, ROUGE, and Cosine Similarity:
```bash
python scripts/metrics_data_gen.py
```

---

## 🎯 How It Works

### **🔍 Document Ingestion & Chunking**
- Extracts text from PDFs.
- Splits text into **semantic chunks** using `RecursiveCharacterTextSplitter`.

### **💾 Embedding & Storage**
- Uses `nomic-embed-text` for **embedding generation**.
- Stores embeddings in **ChromaDB** for **fast retrieval**.

### **🔎 Retrieval & Generation**
- Queries are matched against **ChromaDB** embeddings.
- The best-matched contexts are used to generate responses via **Llama 3.2**.

### **📊 Evaluation**
- Computes **BLEU, ROUGE (1,2,L), Cosine Similarity** between retrieved/generated responses and expected answers.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|--------|-----------|
| `ModuleNotFoundError: No module named 'chromadb'` | Run `pip install chromadb` |
| `Database not updating after adding new PDFs` | **Delete the `chroma/` folder** and rerun `populate_database.py` |
| `Query results are incorrect or empty` | Ensure the database was populated correctly before running `query_data.py` |
| `Ollama model not found` | Run `ollama pull llama3.2` to download the required model |

---

## 🔮 Future Enhancements
- ✅ **API Support**
- ✅ **Web UI Integration**
- ✅ **More Evaluation Metrics**
- ✅ **Fine-Tuning Support**
