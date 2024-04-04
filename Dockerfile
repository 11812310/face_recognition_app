FROM python:3.8.17-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libgl1-mesa-glx libglib2.0-0 && apt-get clean

COPY requirements.txt /app

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu 
RUN pip3 install -r requirements.txt
RUN pip3 install ultralytics --no-deps

COPY . .

EXPOSE 5000

CMD [ "flask", "--app", "recognition_app", "run", "--host", "0.0.0.0"]