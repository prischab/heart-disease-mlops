# Dockerfile — the recipe for packaging this app into a container.
#
# A container is a sealed box with everything the app needs: the right Python,
# the right libraries, and the code. It runs identically on any machine.
#
# Build it:   docker build -t heart-api .
# Run it:     docker run -p 8000:8000 heart-api

# 1. Start from a small official Python image (a clean Linux + Python 3.11)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install dependencies first (this layer is cached, so rebuilds are fast
#    when only your code changes, not your requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code and the trained model into the container
COPY app.py schema.py ./
COPY model/ ./model/

# 5. Tell Docker the app listens on port 8000
EXPOSE 8000

# 6. The command that runs when the container starts
#    (0.0.0.0 makes it reachable from outside the container, not just inside)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]