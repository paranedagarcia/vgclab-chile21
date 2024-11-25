FROM python:3.12-slim
LABEL maintainer="Patricio Araneda"
LABEL email="paraneda@dataclinica.cl"
LABEL version="1.0"
LABEL description="App de Streamlit para el curso de análisis de datos de eventos adversos"
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY  . /app


RUN pip3 install --no-cache-dir  -r requirements.txt

EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Inicio.py", "--server.port=80", "--server.address=0.0.0.0"]
