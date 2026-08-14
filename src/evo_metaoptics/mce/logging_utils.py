"""Logging helpers for run and agent log files."""

import logging
from typing import Optional
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted

def setup_run_logger(log_base_dir: str = "logs", *, run_id: str | None = None) -> Path:
    """
    Create a run directory for organizing logs.
    
    When *run_id* is provided the directory is named ``run_{run_id}`` so that
    the caller (typically a shell entry-point) can share a single identifier
    across workspace and log directories.  When omitted the current timestamp
    is used (backward-compatible default).
    
    Args:
        log_base_dir: Base directory for logs (default: "logs")
        run_id: Optional explicit run identifier.  When ``None``, a
            ``YYYYMMDD_HHMMSS`` timestamp is generated automatically.
        
    Returns:
        Path to the created run directory
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_base_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(
    name: str = "mce",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    console_colors: bool = True,
    run_dir: Optional[Path] = None,
    agent_type: Optional[str] = None,
    iteration: Optional[int] = None,
    sub_iteration: Optional[int] = None,
    minimal_console: bool = False
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files (used if run_dir is None)
        log_level: Logging level (default: INFO)
        console_colors: Whether to use colors in console output
        run_dir: Run directory for organized logging (overrides log_dir if provided)
        agent_type: Agent type for naming ("meta", "base", "eval")
        iteration: Iteration number for naming
        sub_iteration: Sub-iteration number for naming (for online learning batches)
        minimal_console: Whether to use minimal console output
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent propagation to parent loggers to avoid duplicate output
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Determine log directory and file name
    if run_dir is not None:
        log_path = Path(run_dir)
        if agent_type and iteration is not None and sub_iteration is not None:
            log_file = log_path / f"{agent_type}_iter{iteration}_sub{sub_iteration}.log"
        elif agent_type and iteration is not None:
            log_file = log_path / f"{agent_type}_iter{iteration}.log"
        elif agent_type:
            log_file = log_path / f"{agent_type}.log"
        else:
            log_file = log_path / f"{name}.log"
    else:
        log_path = Path(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"session_{timestamp}.log"
    
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler with minimal or full output
    if not minimal_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        if console_colors:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (no colors, full output)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    if not minimal_console:
        logger.info(f"Logging to file: {log_file}")
    
    return logger
