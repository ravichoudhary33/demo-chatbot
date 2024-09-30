# Taipy LLM Chat Demo

<p align="center">
  <img src="media/app_thumbnail.png" alt="A conversation about calibers" width="100%"/>
</p>

[![Watch the video](https://raw.githubusercontent.com/ravichoudhary33/demo-chatbot/tree/develop/media/app_thumbnail.png)](https://raw.githubusercontent.com/ravichoudhary33/demo-chatbot/tree/develop/media/jansamvad_demo_1080p.mov)


A simple app to chat with an LLM which can be used to create any LLM Inference Web Apps using Python only.

This particular app uses OpenAI's GPT-4 API to generate responses to your messages. You can easily change the code to use any other API or model.

## Tutorial


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

2. get the container id of ollama 
```
 docker ps
 ```

3. Download model using ollama container id
 ```
 docker exec -it <ContainerID> ollama run phi
 ```