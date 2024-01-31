# Base image
FROM python:3.10

# Install all required packages to run the model
RUN apt update && apt install --yes ffmpeg libsm6 libxext6
