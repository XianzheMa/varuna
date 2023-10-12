# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN apt-get update --fix-missing  \
    && apt-get install -y build-essential --no-install-recommends

WORKDIR /workspace

COPY ./external/apex ./apex

RUN cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    && cd .. \
    && rm -rf apex

RUN apt install curl lsof vim -y --no-install-recommends \
    && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl


RUN pip install wandb \
    && pip install kubernetes==21.7.0 \
    && pip install datasets \
    && pip install google-cloud-storage==2.10.0


COPY ./varuna ./varuna
COPY ./setup.py ./setup.py
COPY ./tools ./tools
RUN python setup.py install && rm -rf varuna && rm -rf tools
COPY ./examples/gpt ./
