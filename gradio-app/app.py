import gradio as gr
import random
import time
import ollama
from transformers import pipeline
import numpy as np

# history = [["user_msg_1", "bot_msg_1"], ["user_msg_2", "bot_msg_2"]]

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

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
    client = ollama.Client(host='http://ollama:11434')
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
    stream = client.chat(model="phi:latest", stream=True, messages=messages)

    history[-1][1] = ""
    for chunk in stream:
        if chunk["message"]["content"] is not None:
            history[-1][1] += chunk["message"]["content"]
            yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    # msg = gr.Textbox()
    msg = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)
    audio_input = gr.Audio(sources="microphone")
    clear = gr.Button("Clear")

    def user(user_message, history, audio_input):
        print(f"user_msg: {user_message}")
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
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10)
    demo.launch()

