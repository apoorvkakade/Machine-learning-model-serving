FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /usr/src/app

COPY . .

RUN apt-get update

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","model_server.py"]