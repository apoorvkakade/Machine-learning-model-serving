FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /usr/src/app

COPY . .

RUN apt-get update

RUN apt-get install --yes gunicorn

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "fastapi[all]"

EXPOSE 8000

CMD ["gunicorn", "web_server:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]