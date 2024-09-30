import gradio as gr
import random
import time
import ollama
from transformers import pipeline
import numpy as np
import chromadb

# from langdetect import detect
# from translate import Translator

# history = [["user_msg_1", "bot_msg_1"], ["user_msg_2", "bot_msg_2"]]

MODEL = "llama3.1:8b"
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
# hi_2_en_translator = Translator(from_lang="hi", to_lang="en")

rag_cache = {}

chromadb_client = chromadb.HttpClient(host="chromadb-vecdb", port=8000)
collection = chromadb_client.get_collection(name="docs")
ollama_client = ollama.Client(host='http://ollama:11434')

def retrieve_context(prompt):
    if prompt in rag_cache:
        return rag_cache[prompt]
    # generate an embedding for the prompt and retrieve the most relevant doc
    prompt_embedding = ollama_client.embeddings(
        prompt=prompt,
        model=MODEL
    )
    # get relevant document
    rag_contexts = collection.query(
        query_embeddings=[prompt_embedding["embedding"]],
        n_results=2
    )
    # build the rag context string
    rag_context_str = "\n".join(rag_contexts['documents'][0])
    return rag_context_str, rag_contexts


def rag_bot(history):
    # print(f"bot history: {history}")
    client = ollama_client#ollama.Client(host='http://ollama:11434')
    messages = []
    for human, ai in history[:-1]:
        if human:
            messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})

    # get the user prompt from history
    prompt = history[-1][0]
    rag_context_str, rag_contexts = retrieve_context(prompt)
    # print(f"rag_context: {rag_contexts}")
    # append the last user message with rag context
    messages.append(
        {
            'role': 'user',
            'content': f"""Given the following context: {rag_context_str}, 
                        provide a concise and direct response to this prompt: {prompt}.
                        Limit your answer to a maximum of 50 words.
                        Do not provide explanations unless explicitly asked.""",
        }
    )
    
    stream = client.chat(model=MODEL, stream=True, messages=messages)
    history[-1][1] = ""
    for chunk in stream:
        if chunk["message"]["content"] is not None:
            history[-1][1] += chunk["message"]["content"]
            yield history
    

def transcribe(audio):
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def bot(history):
    print(f"bot history: {history}")
    client = ollama_client#ollama.Client(host='http://ollama:11434')
    messages = []
    for human, ai in history[:-1]:
        if human:
            messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})

    # append the last user message
    messages.append(
        {
            "role": "user", "content": history[-1][0]
        }
    )
    stream = client.chat(model=MODEL, stream=True, messages=messages)

    history[-1][1] = ""
    for chunk in stream:
        if chunk["message"]["content"] is not None:
            history[-1][1] += chunk["message"]["content"]
            yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=550)
    # msg = gr.Textbox()
    msg = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)
    audio_input = gr.Audio(sources="microphone")
    clear = gr.Button("Clear")

    def user(user_message, history, audio_input):
        # print(f"user_msg: {user_message}")
        # if user_message['text']:
        #     user_lang = detect(user_message['text'])
        #     print(f"user_lang: {user_lang}")
        #     if user_lang == 'hi':
        #         # translate to eng
        #         user_message['text'] = hi_2_en_translator.translate(user_message['text'])

        if not user_message["text"] and audio_input:
            user_message["text"] = transcribe(audio_input)
            audio_input = None
        
        updated_history = history + [[user_message["text"], None]]
        user_message["text"] = ""
        return user_message, updated_history

    # def bot(history):
    #     bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    #     history[-1][1] = ""
    #     for character in bot_message:
    #         history[-1][1] += character
    #         time.sleep(0.05)
    #         yield history

    msg.submit(user, [msg, chatbot, audio_input], [msg, chatbot], queue=False).then(
        rag_bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10)
    demo.launch()

