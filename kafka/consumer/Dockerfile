FROM python:3.10-slim

WORKDIR /app

COPY ./consumer.py .
COPY ./scaler.pkl .
COPY ../requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "consumer.py"]