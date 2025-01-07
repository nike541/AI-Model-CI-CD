FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Expose the port the app runs on
EXPOSE 8000
EXPOSE 8001

ENV NAME World

# Run the application
CMD ["python", "server.py"]