FROM python:3.13-slim

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

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "src.predictor:app"]
