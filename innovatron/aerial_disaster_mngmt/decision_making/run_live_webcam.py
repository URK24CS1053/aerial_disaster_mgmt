#!/usr/bin/env python3
"""
Quick launch script for live webcam analysis.
Skips the menu and goes directly to webcam perception.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_making.main import run_live_webcam_analysis
from decision_making.logger_config import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SAR SYSTEM - LIVE WEBCAM ANALYSIS")
    print("="*70)
    print("\nStarting live perception from webcam...\n")
    
    logger.info("Starting live webcam analysis")
    
    try:
        success = run_live_webcam_analysis()
        if success:
            print("\n[OK] Live analysis completed successfully")
            logger.info("Live analysis completed successfully")
        else:
            print("\n[ERROR] Live analysis failed - check logs")
            logger.error("Live analysis failed")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
