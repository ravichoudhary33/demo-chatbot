# JanSamvad AI Chatbot

<p align="center">
  <img src="media/app_thumbnail.png" alt="A conversation about calibers" width="100%"/>
</p>

https://github.com/user-attachments/assets/11afcd55-2ab9-486c-972f-4ee2420d0521



A simple app to chat with an LLM which can be used to query Governemnt Schemes.

This particular app uses Ollama API to generate responses to your messages.


## How to Use

1. Clone this repo:

```bash	
git clone https://github.com/ravichoudhary33/demo-chatbot.git
```

**Docker compose method (no api key required) uses ollama locally**
1. Build with docker compose
```
docker-compose up -d --build
```

2. get the container id of ollama by running comamnd `docker ps`
```
 CONTAINER ID   IMAGE                             COMMAND                  CREATED       STATUS                 PORTS                      NAMES
a21426cda171   jupyter/minimal-notebook:latest   "tini -g -- start-no…"   5 hours ago   Up 5 hours (healthy)   0.0.0.0:8888->8888/tcp     jupyter
3c3bcdc419e9   demo-chatbot-gradio_app           "gradio app.py"          5 hours ago   Up 5 hours             0.0.0.0:7860->7860/tcp     demo-chatbot-gradio_app-1
ba2d75b2c2b8   ollama/ollama                     "/bin/ollama serve"      5 hours ago   Up 5 hours             0.0.0.0:11434->11434/tcp   demo-chatbot-ollama-1
bafb2793125a   chromadb/chroma:latest            "/docker_entrypoint.…"   5 hours ago   Up 5 hours             0.0.0.0:8000->8000/tcp     demo-chatbot-chromadb-vecdb-1
 ```

3. Download model using ollama container id, id will change each time when you run docker compose up command
 ```
 docker exec -it ba2d75b2c2b8 ollama run llama3.2:1b
 ```
4. Open jupter notebook at `0.0.0.0:8888` in the browser, enter while asking for password since password is empty and run the `vec_store_util.ipynb` notebook to create collection, you can see the running container and exposed ports by using `docker ps`

 5. Open gradio app at `localhost:7860` in your browser to interact with the app
