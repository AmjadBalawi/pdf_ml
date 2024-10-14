FROM python:3.9-slim

# Set environment variables
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "your_app_module:app", "--host", "0.0.0.0", "--port", "8000"]
