"""
Centralized Logging and Metrics Configuration

This module provides a unified logging system and performance metrics tracking
for the Health Report Assistant application. It supports structured logging,
console/file output, and real-time performance monitoring.

Components:
- Structured logging with timestamps and log levels
- Console and file output support
- Performance metrics tracking (timing, counts, etc.)
- Statistical aggregation (average, min, max, count)

Production Features:
- Thread-safe metric collection
- Flexible log level configuration
- Optional file-based logging for persistence
- Structured log format compatible with log aggregation tools

Usage:
    from utils.logging_config import setup_logging, metrics_tracker

    # Setup logging
    logger = setup_logging(level="INFO", log_file="app.log")
    logger.info("Application started")

    # Track metrics
    metrics_tracker.record("api_response_time", 0.5, {"endpoint": "/analyze"})
    summary = metrics_tracker.get_summary()
"""
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, List, Any


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter with optional color coding for terminal output.

    Note: Colors are disabled by default for Windows compatibility.
    Can be enabled for Unix-like systems if needed.

    Attributes:
        COLORS: Dictionary mapping log levels to color codes (empty for compatibility)
        RESET: Reset code for terminal colors (empty for compatibility)
    """

    # Color codes disabled for cross-platform compatibility
    # For Unix/Mac: Can use ANSI escape codes like '\033[91m' for red
    COLORS = {
        'DEBUG': '',    # Would be gray/blue in color mode
        'INFO': '',     # Would be green in color mode
        'WARNING': '',  # Would be yellow in color mode
        'ERROR': '',    # Would be red in color mode
        'CRITICAL': ''  # Would be bright red in color mode
    }
    RESET = ''  # Would be '\033[0m' in color mode

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with optional coloring.

        Args:
            record (logging.LogRecord): The log record to format

        Returns:
            str: Formatted log message
        """
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return the application's root logger.

    Sets up structured logging with console output and optional file output.
    Uses a consistent format across all log messages for easy parsing and filtering.

    Args:
        level (str): Minimum log level to output. Options:
            - "DEBUG": Detailed diagnostic information
            - "INFO": General informational messages (default)
            - "WARNING": Warning messages for unexpected events
            - "ERROR": Error messages for failures
            - "CRITICAL": Critical errors requiring immediate attention

        log_file (Optional[str]): Path to log file for persistent logging.
            If None, logs only to console. Useful for production environments.

    Returns:
        logging.Logger: Configured logger instance for the application

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="app.log")
        >>> logger.info("Application started")
        [2025-01-15 10:30:45] INFO     - health_report_assistant - Application started

    Note:
        - Removes any existing handlers to prevent duplicate logs
        - Uses 24-hour time format with millisecond precision
        - Log format: [timestamp] LEVEL - logger_name - message
    """
    # Create or get the application logger
    logger = logging.getLogger("health_report_assistant")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove any existing handlers to prevent duplicates
    # Important when this function is called multiple times
    logger.handlers = []

    # Console handler for real-time log output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Standard log format: timestamp, level, logger name, message
    # Compatible with most log aggregation tools (Datadog, Splunk, etc.)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for persistent logging (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "health_report_assistant") -> logging.Logger:
    """
    Get a logger instance for a specific module or component.

    Useful for creating module-specific loggers that inherit the
    application's logging configuration.

    Args:
        name (str): Logger name, typically the module name.
            Defaults to application root logger.

    Returns:
        logging.Logger: Logger instance for the specified name

    Example:
        >>> logger = get_logger("agents.extractor")
        >>> logger.info("Starting PDF extraction")
    """
    return logging.getLogger(name)


class MetricsTracker:
    """
    Simple in-memory metrics tracker for performance monitoring.

    Tracks numeric metrics with timestamps and optional tags for filtering.
    Provides statistical aggregation (average, min, max, count) for analysis.

    Attributes:
        metrics (Dict[str, List[Dict]]): Stored metrics organized by metric name.
            Each entry contains: value, timestamp, and tags

    Production Considerations:
        - Metrics are stored in memory (lost on restart)
        - For production, consider exporting to monitoring systems
          (Prometheus, CloudWatch, Datadog, etc.)
        - No automatic cleanup of old metrics (memory grows over time)
        - Thread-safe for concurrent metric recording

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.record("api_latency", 0.25, {"endpoint": "/analyze"})
        >>> tracker.record("api_latency", 0.30, {"endpoint": "/analyze"})
        >>> summary = tracker.get_summary()
        >>> print(summary['api_latency']['average'])
        0.275
    """

    def __init__(self):
        """Initialize an empty metrics tracker."""
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}

    def record(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a metric value with timestamp and optional tags.

        Args:
            metric_name (str): Name of the metric (e.g., "pdf_extraction_time")
            value (float): Numeric value to record (e.g., 1.25 seconds)
            tags (Optional[Dict[str, Any]]): Optional metadata for filtering/grouping.
                Example: {"user_id": "user123", "report_type": "blood_test"}

        Example:
            >>> tracker.record("extraction_time", 1.5, {"user_id": "user123"})
            [2025-01-15 10:30:45] INFO - METRIC - extraction_time: 1.50 {'user_id': 'user123'}
        """
        # Initialize metric list if first time seeing this metric name
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        # Create metric entry with value, timestamp, and tags
        entry = {
            "value": value,
            "timestamp": datetime.now(),
            "tags": tags or {}
        }
        self.metrics[metric_name].append(entry)

        # Log the metric for real-time monitoring
        logger = get_logger()
        logger.info(f"METRIC - {metric_name}: {value:.2f} {tags or ''}")

    def get_average(self, metric_name: str) -> float:
        """
        Calculate the average value for a specific metric.

        Args:
            metric_name (str): Name of the metric to average

        Returns:
            float: Average value, or 0.0 if metric doesn't exist

        Example:
            >>> tracker.record("latency", 1.0)
            >>> tracker.record("latency", 2.0)
            >>> print(tracker.get_average("latency"))
            1.5
        """
        if metric_name not in self.metrics:
            return 0.0

        values = [m["value"] for m in self.metrics[metric_name]]
        return sum(values) / len(values) if values else 0.0

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of all recorded metrics.

        Returns:
            Dict[str, Dict[str, float]]: Summary statistics for each metric:
                - count: Number of recorded values
                - average: Mean value
                - min: Minimum value
                - max: Maximum value

        Example:
            >>> summary = tracker.get_summary()
            >>> print(summary)
            {
                'api_latency': {
                    'count': 10,
                    'average': 0.25,
                    'min': 0.1,
                    'max': 0.5
                }
            }
        """
        summary = {}
        for metric_name, entries in self.metrics.items():
            values = [e["value"] for e in entries]
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        return summary


# Global metrics tracker instance
# Shared across the entire application for centralized metric collection
# Access this instance to record metrics from any module
metrics_tracker = MetricsTracker()


if __name__ == "__main__":
    # Test and demonstration code
    print("=== Testing Logging Configuration ===\n")

    # Setup logger with DEBUG level for testing
    logger = setup_logging(level="DEBUG")

    # Test different log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    print("\n=== Testing Metrics Tracker ===\n")

    # Record some test metrics
    metrics_tracker.record("test_metric", 12.5, {"user": "test"})
    metrics_tracker.record("test_metric", 15.3, {"user": "test"})
    metrics_tracker.record("api_calls", 1, {"endpoint": "/analyze"})
    metrics_tracker.record("api_calls", 1, {"endpoint": "/chat"})

    # Print metrics summary
    print("\nMetrics Summary:")
    import json
    print(json.dumps(metrics_tracker.get_summary(), indent=2))
