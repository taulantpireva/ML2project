# Use an official Python runtime as a parent image
FROM python:3.12.3-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the backend directory contents into the container at /app
COPY backend/ /app/backend

# Copy the filtered requirements.txt file and install any needed packages specified in it
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the svelte-app directory contents into the container at /app/svelte-app
COPY svelte-app/public /app/svelte-app/public

# Set the working directory
WORKDIR /app/backend

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=/app/backend/app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
