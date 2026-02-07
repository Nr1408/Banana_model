FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# EXPOSE isn't strictly needed for Railway but good for documentation
EXPOSE 8080

# --- THE FIX ---
# We wrap the command in "sh -c" to force the $PORT variable to work
CMD sh -c "uvicorn main.py:app --host 0.0.0.0 --port $PORT"