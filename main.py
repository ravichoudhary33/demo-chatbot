import os
import sys
import time

from taipy.gui import Gui, State, notify
import openai
import ollama
import chromadb

from dotenv import load_dotenv

chromadb_client = None
ollama_client = None
collection = None

client = None
context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI assistant. How can I help you today? "
conversation = {
    "Conversation": ["Who are you?", "Hi! I am JanSamvad AI assistant. How can I help you today?"]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]


def on_init(state: State) -> None:
    """
    Initialize the app.

    Args:
        - state: The current state of the app.
    """
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI assistant. How can I help you today? "
    state.conversation = {
        "Conversation": ["Who are you?", "Hi! I am JanSamvad AI assistant. How can I help you today?"]
    }
    state.current_user_message = ""
    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]
    state.collection = state.chromadb_client.get_collection(name="docs")


def request(state: State, prompt: str) -> str:
    """
    Send a prompt to the JanSamvad AI assistant API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    response = state.client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model="gpt-4-turbo-preview",
    )
    return response.choices[0].message.content


def retrieve_context(state: State, prompt: str) -> str:
    # generate an embedding for the prompt and retrieve the most relevant doc
    prompt_embedding = state.ollama_client.embeddings(
        prompt=prompt,
        model='phi:latest'
    )
    # get relevant document
    rag_contexts = state.collection.query(
        query_embeddings=[prompt_embedding["embedding"]],
        n_results=2
    )
    # build the rag context string
    rag_context_str = "\n".join(rag_contexts['documents'][0])
    return rag_context_str


def ollama_request(state: State, prompt: str) -> str:
    """
    Send a prompt to the ollama API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    # retrieve the rag context first for the input prompt
    rag_context_str = retrieve_context(state, prompt)

    #ollama_client = ollama.Client(host='http://ollama:11434')
    response = state.ollama_client.chat(
        messages=[
            {
                'role': 'user',
                'content': f"Using this data: {rag_context_str}. Respond to this prompt: {prompt}",
            }
        ],
        model='phi:latest',
        stream=False
    )
    answer = response['message']['content'].replace("\n", "")
    if answer in ['', ' ']:
        answer = "Please add more specific detail about the query!"
    
    return answer

# send message v2 version for streaming
def send_message_v2(state: State) -> None:
    
    notify(state, "info", "Sending message and streaming...")
    # update context mehtod 
    state.context += f"Human: \n {state.current_user_message}\n\n AI:"
    prompt = state.context
    # answer = request(state, state.context).replace("\n", "")
    rag_context_str = retrieve_context(state, prompt)
    # use the rag context generate llm output in stream fashion
    stream = state.ollama_client.chat(
        messages=[
            {
                'role': 'user',
                'content': f"Using this data: {rag_context_str}. Respond to this prompt: {prompt}",
            }
        ],
        model='phi:latest',
        stream=True
    )
    # using stream generator stream the result
    conv = state.conversation._dict.copy()
    # append current user message and empty string as last message as dummy
    conv["Conversation"] += [state.current_user_message, '']
    answer = ""
    for chunk in stream:
        msg_chunks = chunk['message']['content']
        answer += msg_chunks
        #print(msg_chunks, end='', flush=True)
        conv["Conversation"][-1] = [answer] # update the last message with the stream chunk message
        
    state.context += answer
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    # update contex is complete
    # set the current user message to empty string
    state.current_user_message = ""
    state.conversation = conv
    notify(state, "success", "Stream response received!")

def update_context(state: State) -> None:
    """
    Update the context with the user's message and the AI's response.

    Args:
        - state: The current state of the app.
    """
    state.context += f"Human: \n {state.current_user_message}\n\n AI:"
    # answer = request(state, state.context).replace("\n", "")
    answer = ollama_request(state, state.context)
    state.context += answer
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    return answer


def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the context.

    Args:
        - state: The current state of the app.
    """
    notify(state, "info", "Sending message...")
    answer = update_context(state)
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.current_user_message = ""
    state.conversation = conv
    notify(state, "success", "Response received!")


def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "gpt_message"


def on_exception(state, function_name: str, ex: Exception) -> None:
    """
    Catches exceptions and notifies user in Taipy GUI

    Args:
        state (State): Taipy GUI state
        function_name (str): Name of function where exception occured
        ex (Exception): Exception
    """
    notify(state, "error", f"An error occured in {function_name}: {ex}")


def reset_chat(state: State) -> None:
    """
    Reset the chat by clearing the conversation.

    Args:
        - state: The current state of the app.
    """
    state.past_conversations = state.past_conversations + [
        [len(state.past_conversations), state.conversation]
    ]
    state.conversation = {
        "Conversation": ["Who are you?", "Hi! I am JanSamvad AI assistant. How can I help you today?"]
    }


def tree_adapter(item: list) -> [str, str]:
    """
    Converts element of past_conversations to id and displayed string

    Args:
        item: element of past_conversations

    Returns:
        id and displayed string
    """
    identifier = item[0]
    if len(item[1]["Conversation"]) > 3:
        return (identifier, item[1]["Conversation"][2][:50] + "...")
    return (item[0], "Empty conversation")


def select_conv(state: State, var_name: str, value) -> None:
    """
    Selects conversation from past_conversations

    Args:
        state: The current state of the app.
        var_name: "selected_conv"
        value: [[id, conversation]]
    """
    state.conversation = state.past_conversations[value[0][0]][1]
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI assistant. How can I help you today? "
    for i in range(2, len(state.conversation["Conversation"]), 2):
        state.context += f"Human: \n {state.conversation['Conversation'][i]}\n\n AI:"
        state.context += state.conversation["Conversation"][i + 1]
    state.selected_row = [len(state.conversation["Conversation"]) + 1]


past_prompts = []

page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# JanSamvad **AI**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
### Previous activities ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":
    load_dotenv()

    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    chromadb_client = chromadb.HttpClient(host="chromadb-vecdb", port=8000)
    ollama_client = ollama.Client(host='http://ollama:11434')

    Gui(page).run(debug=True, dark_mode=True, use_reloader=True, title="💬 Taipy Chat")
