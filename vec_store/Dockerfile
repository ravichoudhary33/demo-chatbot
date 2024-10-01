FROM python:3.11

WORKDIR /app

# Install application dependencies.
COPY ./vec_store/requirements.txt .
RUN pip install -r requirements.txt

# Copy the application source code.
COPY ./vec_store/main.py .

CMD ["python", "main.py"]