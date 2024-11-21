# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
