import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOGFILE = Path("logs") / "mmdfled.log"
DEFAULT_LOGFILE.parent.mkdir(exist_ok=True)

def setup_logger(
    name: str,
    level: int = logging.INFO,
    logfile: Path | str = DEFAULT_LOGFILE,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    fh = RotatingFileHandler(
        filename=str(logfile),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh_formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fh_formatter)
    logger.addHandler(ch)

    return logger
