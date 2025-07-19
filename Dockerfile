# Step 1: Use an official Python runtime as a parent image
# NOTE: Change '3.9' to match the Python version in your runtime.txt file
FROM python:3.11.0-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements file and install dependencies
# This is done in a separate step to leverage Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the rest of your application code into the container
COPY . .

# Step 5: Define the command to run your application
# This assumes your app.py starts a web server (e.g., Flask, FastAPI).
# If your app listens on a different port, you will need to adjust the 'docker run' command later.
CMD ["python", "app.py"]