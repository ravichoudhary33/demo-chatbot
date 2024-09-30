import ollama
import chromadb
from llama_index.core import SimpleDirectoryReader
import concurrent.futures


MODEL = "llama3.1:8b"

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

# async def process_documents_async(documents, ollama_client, collection, MODEL):
#     tasks = [process_document(doc, ollama_client, collection, MODEL) for doc in documents]
#     await asyncio.gather(*tasks)


# # Usage:
# async def main():
#     chromadb_client = await chromadb.HttpClient(host="chromadb-vecdb", port=8000)
#     ollama_client = await ollama.Client(host='http://ollama:11434')

#     documents = SimpleDirectoryReader("/app/data/tmp").load_data()
#     try:
#         collection = await chromadb_client.create_collection(name="docs")
#     except Exception as e:
#         await chromadb_client.delete_collection(name="docs")
#         collection = await chromadb_client.create_collection(name="docs")
    
#     await process_documents_async(documents, ollama_client, collection, MODEL)


# if __name__ == "__main__":
#     asyncio.run(main())


def create_collection_vecdb(chromadb_client, ollama_client):

    documents = SimpleDirectoryReader("/app/data/tmp").load_data()
    try:
        collection = chromadb_client.create_collection(name="docs")
    except Exception as e:
        chromadb_client.delete_collection(name="docs")
        collection = chromadb_client.create_collection(name="docs")

    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_document, doc, ollama_client, MODEL, collection) for doc in documents]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Wait for the result to avoid exceptions

    # store each document in a vector embedding database
    # for i, doc in enumerate(documents):
    #     # get the document id and text
    #     doc_id = doc.id_
    #     doc_text = doc.get_text()
    #     doc_metadata = doc.metadata
    
    #     response = ollama_client.embeddings(model=MODEL, prompt=doc_text)
    #     embedding = response["embedding"]
    #     collection.add(
    #         ids=[doc_id],
    #         embeddings=[embedding],
    #         documents=[doc_text],
    #         metadatas=doc_metadata
    #     )
    
    print(f"Collection created!!!")


if __name__ == "__main__":

    chromadb_client = chromadb.HttpClient(host="chromadb-vecdb", port=8000)
    ollama_client = ollama.Client(host='http://ollama:11434')
    create_collection_vecdb(chromadb_client, ollama_client)
