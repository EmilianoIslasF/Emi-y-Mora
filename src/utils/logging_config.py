"""
Configuración centralizada de logging.
Escribe logs en consola y en artifacts/logs/.
"""

import logging
from datetime import datetime
from pathlib import Path

from src.utils.paths import ARTIFACTS_DIR


def setup_logger(script_name: str) -> logging.Logger:
    """
    Crea un logger con:
    - salida a consola
    - archivo en artifacts/logs/<script>_<timestamp>.log
    """
    log_dir = ARTIFACTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Evitar duplicar handlers si se corre varias veces
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logger inicializado. Archivo: {log_file}")
    return logger
