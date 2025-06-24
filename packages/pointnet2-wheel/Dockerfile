FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken && \
    apt-get -y update && apt install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils curl && \
    rm -rf /var/lib/apt/lists/* 

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 

RUN python3.11 -m pip install torch==2.0.1
RUN python3.11 -m pip install -U setuptools==69.*