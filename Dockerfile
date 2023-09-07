FROM python:3

WORKDIR /app

COPY requirements.txt .

EXPOSE 8888

RUN pip install --no-cache-dir -r requirements.txt
