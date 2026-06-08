from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_console: Console | None = None


def get_console() -> Console:
    """Return the shared Rich :class:`~rich.console.Console` instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


class RichMessageFormatter(logging.Formatter):
    """Formatter that colour-codes log records by severity level."""

    _LEVEL_STYLES: dict[int, str] = {
        logging.DEBUG: "cyan",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "bold red",
        logging.CRITICAL: "bold white on red",
    }

    def format(self, record: logging.LogRecord) -> str:
        style = self._LEVEL_STYLES.get(record.levelno, "default")
        original_levelname = record.levelname
        record.levelname = f"[bold]{record.levelname:<15}[/bold]"
        message = super().format(record)
        record.levelname = original_levelname
        return f"[{style}]{message}[/{style}]"


def setup_logger(name: str, level: str = "INFO", log_file: Path | str | None = None) -> None:

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    _FMT = "%(levelname)-15s %(asctime)s | %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    console_handler = RichHandler(
        markup=True,
        console=get_console(),
        show_time=False,
        show_level=False,
        rich_tracebacks=True,
    )
    console_handler.setFormatter(RichMessageFormatter(fmt=_FMT, datefmt=_DATE_FMT))
    console_handler.setLevel(level.upper())
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(level.upper())
        file_handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT))
        logger.addHandler(file_handler)

    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)