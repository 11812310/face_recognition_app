FROM python:3.8.17-slim-bookworm

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu 
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "flask", "--app", "recognition_app", "run", "--host", "0.0.0.0"]