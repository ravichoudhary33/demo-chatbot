FROM python:3.8-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies.
COPY ./gradio-app/requirements.txt .
RUN pip install -r requirements.txt

# Copy the application source code.
COPY ./gradio-app/app.py .
COPY ./gradio-app/models.py .

# RUN pip install --no-cache-dir gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["gradio", "app.py"]
# CMD ["python", "app.py"]