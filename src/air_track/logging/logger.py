import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

# Shared console instance for consistent output
_console = None


def get_console() -> Console:
    """Get the shared console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


class RichMessageFormatter(logging.Formatter):
    """
    A custom formatter that applies rich styles based on log level.
    """

    level_styles = {
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "bold red",
        logging.CRITICAL: "bold white on red",
        logging.DEBUG: "cyan",
    }

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # Pick style based on level
        style = self.level_styles.get(record.levelno, "default")

        # Apply bold to levelname
        original_levelname = record.levelname
        record.levelname = f"[bold]{record.levelname:<15}[/bold]"

        # Format message using the base formatter
        message = super().format(record)

        # Restore original levelname
        record.levelname = original_levelname

        # Apply color to entire message
        return f"[{style}]{message}[/{style}]"


def setup_logger(
    level: str = "INFO",
    log_file: str | None = None,
    verbose: bool = False,
) -> None:
    """
    Sets up a custom logger with a console and file handler.
    """

    logger = logging.getLogger("air_track")

    if logger.hasHandlers():
        logger.handlers.clear()

    # Decide the base log level
    base_level = (
        logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    )
    logger.setLevel(base_level)

    if verbose:
        # Detailed format for debugging
        frmt = (
            "%(levelname)-15s %(asctime)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
    else:
        # Simple format for normal use
        frmt = "%(levelname)-15s %(asctime)s | %(message)s"

    datefmt = "%Y-%m-%d %H:%M:%S"

    # --- Console Handler ---
    console = get_console()
    console_handler = RichHandler(
        markup=True,
        console=console,
        show_time=False,
        show_level=False,
        rich_tracebacks=True,
    )
    formatter = RichMessageFormatter(fmt=frmt, datefmt=datefmt)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level.upper())
    logger.addHandler(console_handler)

    # --- File Handler ---
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level.upper())
        plain_formatter = logging.Formatter(fmt=frmt, datefmt=datefmt)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"air_track.{name}")
    return logging.getLogger("air_track")
