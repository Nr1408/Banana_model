FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# EXPOSE isn't strictly needed for Railway but good for documentation
EXPOSE 8080

# --- THE FIX IS HERE ---
# No brackets, no quotes. This lets $PORT work correctly.
CMD uvicorn main.py:app --host 0.0.0.0 --port $PORT