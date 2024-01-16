# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements-inference.txt

# Define environment variables for MLflow settings
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:8081"
ENV MODEL_NAME="mediwatch-prediction"

# Download the model from MLflow to be packaged as part of the docker image. 
# This avoids reliance on MLflow server as the application scales in production.
RUN python download_and_save_model.py

# Expose the port that FastAPI will run on
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
