FROM python:3.8.17-slim-bookworm

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libgl1-mesa-glx libglib2.0-0 && apt-get clean

COPY requirements.txt /app

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu 
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir ultralytics --no-deps

COPY . .

EXPOSE 5000

CMD exec gunicorn --bind:5000 --workers 1 --threads 8 --timeout 0 app:recognition_app