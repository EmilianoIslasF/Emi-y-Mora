FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir \
    joblib \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    PyYAML

CMD ["python", "-m", "src.inference"]