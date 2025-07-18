FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for PDF output
RUN mkdir -p WEB_KETEBALAN_PER_100M WEB_KETINGGIAN_PER_100M WEB_DEVIASI_PER_100M

# Expose port
EXPOSE $PORT

# Start command
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
