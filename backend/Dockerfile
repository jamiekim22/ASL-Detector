FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgtk-3-0 \
    python3-tk \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../../frontend/templates /app/frontend/templates
COPY ../../frontend/static /app/frontend/static

# Create empty directory for conditional mounting
RUN mkdir -p /tmp/empty

# Copy platform detection script
# COPY scripts/detect_camera.py .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "backend/app/main.py"]