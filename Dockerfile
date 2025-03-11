# Use Python 3.10 instead of 3.9
FROM python:3.10

WORKDIR /app
COPY . /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose the FastAPI port
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

