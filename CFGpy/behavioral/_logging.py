import logging
from pathlib import Path


def build_pipeline_logger(output_filename: str, verbose: bool) -> logging.Logger:
    logger_name = f"CFGpy.pipeline.{Path(output_filename).stem}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if logger is reused
    if logger.handlers:
        return logger

    log_path = f"{output_filename}.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


class HasLogger:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("CFGpy.null")

    def log_info(self, msg: str) -> None:
        self.logger.info(msg)

    def log_warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def log_error(self, msg: str) -> None:
        self.logger.error(msg)