FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        flask \
        gunicorn \
        joblib \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        PyYAML

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "serve.py"]