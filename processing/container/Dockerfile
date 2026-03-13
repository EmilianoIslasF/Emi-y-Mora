FROM python:3.11-slim

WORKDIR /opt/program

COPY . /opt/program

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        pandas \
        numpy \
        scikit-learn
