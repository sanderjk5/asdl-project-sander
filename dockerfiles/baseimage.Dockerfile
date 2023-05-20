# The added dockerfile which works without nvidia cuda.
FROM python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN apt-get -y update; apt-get -y install curl

RUN curl -fsSLO https://get.docker.com/builds/Linux/x86_64/docker-17.04.0-ce.tgz \
  && tar xzvf docker-17.04.0-ce.tgz \
  && mv docker/docker /usr/local/bin \
  && rm -r docker docker-17.04.0-ce.tgz

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade wheel
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

RUN pip install protobuf==3.20.1


RUN mkdir -p /src/buglab
RUN mkdir -p /data/targetDir

WORKDIR /src/
ENV PYTHONPATH=/src/

RUN pip install --upgrade --no-cache-dir py-spy


COPY buglab /src/buglab
WORKDIR /home/buglab
COPY ./package_list.txt package_list.txt

CMD ["python", "-m", "buglab.controllers.staticdatasetextractor", "package_list.txt", "/home/sanderj/projects/neurips21-self-supervised-bug-detection-and-repair/target"]