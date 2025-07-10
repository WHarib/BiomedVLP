FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face cache directory and ensure permissions for non-root
RUN mkdir -p /app/hf_home \
    && chmod -R 777 /app/hf_home

ENV HF_HOME=/app/hf_home

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
