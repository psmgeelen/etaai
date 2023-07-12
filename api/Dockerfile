#https://github.com/svx/poetry-fastapi-docker/blob/main/pyproject.toml
# Dockerfile
# Uses multi-stage builds requiring Docker 17.05 or higher
# See https://docs.docker.com/develop/develop-images/multistage-build/

# Creating a python base with shared environment variables
FROM python:3.9.17-slim-bullseye
SHELL ["/bin/bash", "-c"]
RUN apt-get update
RUN apt-get install gpg curl git -y
# install google coral drivers
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main"| tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install libedgetpu1-std -y

RUN mkdir /api_app
WORKDIR /api_app
COPY /api /api_app

ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-root

CMD [ "uvicorn", "main:app", "--reload"]