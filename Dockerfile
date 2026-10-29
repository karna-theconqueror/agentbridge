# Lightweight Python image – 3.11 is recommended for the latest asyncio features
FROM python:3.11-slim

# Set the work directory inside the container
WORKDIR /app

# Copy all project files into the container (.dockerignore excludes .git/.env)
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Hosts (Render, Fly, HF Spaces…) assign their own port via $PORT.
# Default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

# Start the Uvicorn server when the container boots
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

