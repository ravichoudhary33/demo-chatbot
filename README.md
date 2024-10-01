# JanSamvad AI Chatbot

<p align="center">
  <img src="media/app_thumbnail.png" alt="A conversation about calibers" width="100%"/>
</p>

https://github.com/user-attachments/assets/11afcd55-2ab9-486c-972f-4ee2420d0521



A simple app to chat with an LLM which can be used to query Governemnt Schemes.

This particular app uses Ollama API to generate responses to your messages.

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
