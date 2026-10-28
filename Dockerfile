# Lightweight Python image – 3.11 is recommended for the latest asyncio features
FROM python:3.11-slim

# Set the work directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the API will use
EXPOSE 8000

# Start the Uvicorn server when the container boots
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
