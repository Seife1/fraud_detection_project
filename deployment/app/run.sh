#!/bin/bash

# Define variables
IMAGE_NAME="fraud-detection-model"
CONTAINER_PORT=5000
HOST_PORT=5000

# Function to build Docker image
build_image() {
    echo "Building Docker image..."
    sudo docker build -t $IMAGE_NAME .
}

# Function to run Docker container
run_container() {
    echo "Running Docker container..."
    sudo docker run -p $HOST_PORT:$CONTAINER_PORT $IMAGE_NAME
}

# Check if the model file exists
MODEL_PATH="best_models/gradient_boosting_fraud_data_best_model.pkl"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' does not exist."
    exit 1
fi

# Execute build and run functions
build_image
run_container
