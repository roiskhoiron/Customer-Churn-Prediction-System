# Use official Python 3.10 slim image (matches CI environment)
FROM python:3.10-slim

WORKDIR /app

# Copy only requirement files first for caching
COPY requirements.txt ./
COPY membangun_model/requirements.txt ./membangun_model_requirements.txt
COPY MLProject/requirements.txt ./MLProject_requirements.txt

# Install core deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt && \
    pip install -r membangun_model_requirements.txt && \
    pip install -r MLProject_requirements.txt

# Copy the entire project
COPY . .

# Expose port for possible inference service (optional)
EXPOSE 8000

# Default command (placeholder – replace with Inference.py when ready)
CMD ["python", "membangun_model/modelling.py"]
