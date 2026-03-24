import subprocess
import sys

if __name__ == "__main__":
    # SageMaker puede invocar el contenedor como: docker run image serve
    # Ignoramos cualquier argumento extra y arrancamos Gunicorn.
    cmd = [
        "gunicorn",
        "--bind", "0.0.0.0:8080",
        "src.predictor:app",
    ]
    raise SystemExit(subprocess.call(cmd))