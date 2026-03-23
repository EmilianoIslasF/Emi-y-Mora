FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/ml/processing/code

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY conftest.py ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system .

# Sin CMD ni ENTRYPOINT.
# SageMaker ScriptProcessor va a ejecutar el script explícitamente,
# por ejemplo:
# python /opt/ml/processing/code/src/prep.py
# python /opt/ml/processing/code/src/evaluate.py