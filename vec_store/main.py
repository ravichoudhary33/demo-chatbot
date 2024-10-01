import ollama
import chromadb
from llama_index.core import SimpleDirectoryReader
import concurrent.futures
import asyncio
import hashlib

import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

MODEL = "llama3.2:1b" #"llama3.1:8b" 
MODEL_HASH = hashlib.sha256(MODEL.encode()).hexdigest()[-5:]
COLLECTION_NAME = f"docs_{MODEL_HASH}"

def pull_model(ollama_client, model=MODEL):
    status = ollama_client.pull(model)
    return status

def process_document(doc, ollama_client, MODEL, collection):
    doc_id = doc.id_
    doc_text = doc.get_text()
    doc_metadata = doc.metadata

    response = ollama_client.embeddings(model=MODEL, prompt=doc_text)
    embedding = response["embedding"]

    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[doc_text],
        metadatas=doc_metadata
    )


def create_collection_vecdb(chromadb_client, ollama_client, concurrency=True):

    documents = SimpleDirectoryReader("/app/data").load_data()
    try:
        collection = chromadb_client.create_collection(name=COLLECTION_NAME)
    except Exception as e:
        chromadb_client.delete_collection(name=COLLECTION_NAME)
        collection = chromadb_client.create_collection(name=COLLECTION_NAME)

    if concurrency:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_document, doc, ollama_client, MODEL, collection) for doc in documents]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Wait for the result to avoid exceptions
    else:
        # store each document in a vector embedding database
        for i, doc in enumerate(documents):
            # get the document id and text
            doc_id = doc.id_
            doc_text = doc.get_text()
            doc_metadata = doc.metadata
        
            response = ollama_client.embeddings(model=MODEL, prompt=doc_text)
            embedding = response["embedding"]
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=doc_metadata
            )


if __name__ == "__main__":
    chromadb_client = chromadb.HttpClient(host="chromadb-vecdb", port=8000)
    ollama_client = ollama.Client(host='http://ollama:11434')

    # first pull the model
    logging.info(f"pulling model: {MODEL}")
    res = pull_model(model=MODEL, ollama_client=ollama_client)
    logging.info(f"model pull status: {res}")

    if res['status'] == 'success':
        logging.info(f"creating vector db of the document for rag...")
        create_collection_vecdb(chromadb_client, ollama_client)
        logging.info(f"successfully created vec db!")

    
