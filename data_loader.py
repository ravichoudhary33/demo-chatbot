import ollama
import chromadb
from llama_index.core import SimpleDirectoryReader


MODEL = "phi:latest"

def create_collection_vecdb(chromadb_client, ollama_client):

    documents = SimpleDirectoryReader("/app/data/tmp").load_data()
    try:
        collection = chromadb_client.create_collection(name="docs")
    except Exception as e:
        chromadb_client.delete_collection(name="docs")
        collection = chromadb_client.create_collection(name="docs")

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
    
    print(f"Collection created!!!")


if __name__ == "__main__":

    chromadb_client = chromadb.HttpClient(host="chromadb-vecdb", port=8000)
    ollama_client = ollama.Client(host='http://ollama:11434')
    create_collection_vecdb(chromadb_client, ollama_client)
