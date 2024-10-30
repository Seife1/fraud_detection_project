# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && \
    apt-get install -y build-essential gcc && \
    apt-get clean

# Copy the best_models directory
COPY best_models /app/best_models

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create the logs directory
RUN mkdir -p logs && chmod 777 logs

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run serve_model.py when the container launches
CMD ["python", "serve_model.py"]
