# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
COPY . .

# Expose the backend port
EXPOSE 8000

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0"]
