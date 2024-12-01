FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=America/Los_Angeles

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    git-lfs

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
ENV PYTHONPATH /app/

CMD streamlit run --server.port 8000 --server.address 0.0.0.0 ai_search_demo/ui.py
