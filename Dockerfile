FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# Ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "bot.py"]
