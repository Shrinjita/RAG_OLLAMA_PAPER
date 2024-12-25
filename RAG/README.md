# rag-tutorial-v2

The main idea is that we put any pdf that we want to read in the data directory.
From there, we add it to the database using the populate_database.py script.
Then, we can use the RAG model to answer questions about the text.

To use the RAG mode, run python query_data.py "this is your query"