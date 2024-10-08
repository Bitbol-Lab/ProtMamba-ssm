# Using an official Python runtime with CUDA support as a parent image (https://hub.docker.com/r/nvidia/cuda/)
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN groupadd rcp-runai-upbitbol --gid 80104
# TO CHANGE IF NEEDED
RUN useradd -m -U -s /bin/bash -G rcp-runai-upbitbol -u 268550 malbrank
RUN apt-get update && apt-get install -y git

# Set the working directory: TO CHANGE IF NEEDED
USER malbrank

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1
#install from requirements.txt
COPY requirements.txt /home/malbrank/requirements.txt
#apt get git
RUN pip install --no-cache-dir -r /home/malbrank/requirements.txt
RUN pip install transformers[torch]

WORKDIR /home/malbrank
