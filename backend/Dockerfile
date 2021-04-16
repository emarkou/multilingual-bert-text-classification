FROM python:3.6-slim

WORKDIR /app

RUN apt-get update

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install http://download.pytorch.org/whl/cpu/torch-1.8.1-cp36-cp36m-manylinux1_x86_64.whl
RUN pip install torch==1.8.1
COPY models ./models

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
