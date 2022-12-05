FROM python:3.9-slim

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt

#RUN zenml stack set default
#RUN zenml init