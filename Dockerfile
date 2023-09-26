# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN apt-get update --fix-missing  \
    && apt-get install -y build-essential --no-install-recommends \
    && apt-get install -y git

WORKDIR /workspace

COPY ./install_script.sh ./install_script.sh

RUN chmod +x ./install_script.sh \
    && ./install_script.sh \
    && rm ./install_script.sh

COPY ./examples/gpt ./
