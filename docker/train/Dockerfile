FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/ml/code

WORKDIR /opt/ml/code

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY conftest.py ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system .

ENTRYPOINT ["python", "-m", "src.train"]