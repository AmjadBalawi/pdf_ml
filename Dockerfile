# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Create a virtual environment (optional)
RUN python -m venv /opt/venv

# Set the PATH to the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", 8000]
