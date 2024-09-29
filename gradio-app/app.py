import gradio as gr
from transformers import pipeline
import numpy as np
import random
import time
import ollama

model = "phi:latest"

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def predict(message, history, system_prompt, audio_input):

    if len(message) < 1:
        message = audio_input #transcribe(audio_input)
        print(f"audio_input: {message}")

    convo_history = []
    for human, assistant in history:
        convo_history.append({"role": "user", "content": human })
        convo_history.append({"role": "assistant", "content":assistant})
    
    convo_history.append({"role": "user", "content": message})
    print(f"convo_history: {convo_history}")
    
    ollama_client = ollama.Client(host='http://ollama:11434')
    response = ollama_client.chat(model=model, stream=True, messages=convo_history)

    partial_message = ""
    for chunk in response:
        partial_message = partial_message + chunk["message"]["content"]
        yield partial_message
    
    #print(f"partial_message: {partial_message}")

with gr.Blocks() as demo:
    system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
    # audio_output = gr.Textbox(label="Transcribed Audio")
    # audio_interface = gr.Interface(
    #     transcribe,
    #     gr.Audio(sources="microphone"),
    #     outputs=audio_output,
    # )
    audio_input = gr.Audio(sources="microphone")
    b1 = gr.Button("Recognize Speech")
    audio_input_text = gr.Textbox()
    b1.click(transcribe, inputs=audio_input, outputs=audio_input_text)

    gr.ChatInterface(
        predict, additional_inputs=[system_prompt, audio_input_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)