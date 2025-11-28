# 1) Python Basis-Image
FROM python:3.10-slim

# 2) Arbeitsverzeichnis setzen
WORKDIR /app

# 3) Projektdateien in den Container kopieren
COPY . /app

# 4) Abhängigkeiten installieren
RUN pip install --upgrade pip
RUN pip install -e .

# 5) Standardbefehl
CMD ["python", "-c", "from exercise_pkg.pipeline import data_preparation, train; print('Docker läuft!')"]
