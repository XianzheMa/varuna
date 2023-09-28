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

COPY ./varuna ./varuna
COPY ./setup.py ./setup.py
COPY ./tools ./tools
RUN python setup.py install && rm -rf varuna && rm -rf tools
COPY ./examples/gpt ./
