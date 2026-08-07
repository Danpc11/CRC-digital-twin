# CRC-digital-twin -- imagen reproducible
#
# IMPORTANTE: este Dockerfile vive en docker/ pero el contexto de build
# es la RAIZ del repositorio (por eso los COPY son 'src/', no '../src/').
# Siempre construir desde la raiz con -f:
#
#   docker build -f docker/Dockerfile -t coloq:latest .
#
# Correr la suite de regresion (verificacion de que la imagen esta sana):
#   docker run --rm coloq:latest test
#
# Correr la demo completa sobre datos sinteticos (sin credenciales):
#   docker run --rm -v "$(pwd)/results:/app/results_demo" coloq:latest demo
#
# Lanzar la app web en http://localhost:8501 :
#   docker run --rm -p 8501:8501 coloq:latest app
#
# Trabajar con datos propios (montar el directorio local de datos):
#   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" \
#     coloq:latest calibrate --input data/mi_cohorte.tsv --output results/
#
# O mas simple, con compose (tambien desde la raiz):
#   docker compose -f docker/docker-compose.yml run --rm coloq test

FROM python:3.12-slim

LABEL org.opencontainers.image.title="ColoQ / crc-digital-twin"
LABEL org.opencontainers.image.description="Gemelo digital mecanicista de cancer colorrectal para paneles RT-qPCR"
LABEL org.opencontainers.image.source="https://github.com/Danpc11/CRC-digital-twin"
LABEL org.opencontainers.image.licenses="MIT"

# Dependencias de sistema minimas: build-essential es necesario para
# compilar algunas ruedas cientificas que no traen wheel para slim.
# Se elimina despues de instalar para no inflar la imagen final.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY tests/ ./tests/
COPY cli.py app.py run_pipeline.py ./
COPY README.md PROJECT_STATUS.md CHANGELOG.md LICENSE ./

# Directorios de trabajo -- montables como volumenes desde el host
RUN mkdir -p data results figures

EXPOSE 8501

# Verificacion en tiempo de build: si la suite falla, la imagen no se
# construye. Evita publicar una imagen rota.
RUN python3 -m pytest tests/ -q

ENTRYPOINT ["python3", "cli.py"]
CMD ["--help"]
