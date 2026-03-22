# Base Python 3.11.14 image
FROM python:3.11.14-slim

# Working directory inside container
WORKDIR /app

# Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

# Streamlit port
EXPOSE 8501

# Start Streamlit application
CMD ["streamlit", "run", "app/ui_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=localhost"]
