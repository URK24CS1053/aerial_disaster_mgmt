"""
Logging configuration for the SAR victim detection system.
Provides centralized logging across all modules.
"""

import logging
import logging.handlers
import os
from datetime import datetime


# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Log file paths
LOG_FILE = os.path.join(LOGS_DIR, "sar_system.log")
ERROR_LOG_FILE = os.path.join(LOGS_DIR, "errors.log")


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a configured logger for a specific module.
    
    Args:
        module_name: Name of the module (e.g., 'fusion_engine')
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Console Handler - INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File Handler - DEBUG and above (detailed)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # Error File Handler - ERROR and above
        error_handler = logging.FileHandler(ERROR_LOG_FILE)
        error_handler.setLevel(logging.ERROR)
        error_format = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s\nDetails: %(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_format)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
    
    return logger


def log_system_info():
    """Log system initialization information"""
    logger = get_logger("system")
    logger.info("="*70)
    logger.info("SAR Victim Detection System Initialized")
    logger.info(f"Logs directory: {LOGS_DIR}")
    logger.info(f"Main log file: {LOG_FILE}")
    logger.info(f"Error log file: {ERROR_LOG_FILE}")
    logger.info("="*70)


# Initialize system logging
log_system_info()
