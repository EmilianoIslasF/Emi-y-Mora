FROM python:3.13-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        joblib \
        PyYAML

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "container/train-model/train"]
