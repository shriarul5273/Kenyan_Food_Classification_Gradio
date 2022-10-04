FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip && pip3 install --upgrade pip
RUN mkdir /app
COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]