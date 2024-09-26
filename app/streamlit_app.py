import streamlit as st
import ollama
import chromadb

st.title("JanSamvad AI Assistant")

ollama_client = ollama.Client(host='http://ollama:11434')

chromadb_client = chromadb.HttpClient(host="chromadb-vecdb", port=8000)
collection = chromadb_client.get_collection(name="docs")


# handles stream response back from LLM
def stream_parser(stream):
    for chunk in stream:
        yield chunk['message']['content']

def retrieve_context(prompt) -> str:
    # generate an embedding for the prompt and retrieve the most relevant doc
    prompt_embedding = ollama_client.embeddings(
        prompt=prompt,
        model='phi:latest'
    )
    # get relevant document
    rag_contexts = collection.query(
        query_embeddings=[prompt_embedding["embedding"]],
        n_results=2
    )
    # build the rag context string
    rag_context_str = "\n".join(rag_contexts['documents'][0])
    return rag_context_str


if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "phi:latest"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    rag_context_str = retrieve_context(prompt)
    with st.chat_message("assistant"):
        stream = ollama_client.chat(
            model=st.session_state["ollama_model"],
            messages = [
                {
                    'role': 'user',
                    'content': f"Using this data: {rag_context_str}. Respond to this prompt in no more then 50 words: {prompt}",
                }
            ],
            # messages=[
            #     {"role": m["role"], "content": m["content"]}
            #     for m in st.session_state.messages
            # ],
            stream=True,
        )
        response = st.write_stream(stream_parser(stream))
    st.session_state.messages.append({"role": "assistant", "content": response})

