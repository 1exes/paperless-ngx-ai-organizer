FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY paperless_organizer.py .
COPY web.py .
COPY taxonomy_tags.json .
COPY .env.example .

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data

VOLUME /data

ENTRYPOINT ["python", "paperless_organizer.py", "--autopilot"]
