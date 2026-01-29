#!/usr/bin/env python3
"""
Quick launch script for test scenarios (hardcoded data).
Perfect for testing without hardware.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_making.main import (
    print_header,
    show_system_info,
    show_architecture,
    show_features,
    run_model_evaluation,
    run_victim_scenarios
)
from decision_making.logger_config import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # Print system header
    print_header("SAR VICTIM DETECTION & URGENCY SCORING SYSTEM (REFACTORED)")
    
    logger.info("="*80)
    logger.info("Starting SAR System - Test Scenario Mode")
    logger.info("="*80)
    
    # Show system info
    show_system_info()
    
    # Show architecture
    show_architecture()
    
    # Show features
    show_features()
    
    # Run tests and scenarios
    print_header("RUNNING SYSTEM TESTS & DEMONSTRATIONS")
    
    try:
        results = {
            "model_evaluation": run_model_evaluation(),
            "victim_scenarios": run_victim_scenarios(),
        }
        
        # Final summary
        print_header("SYSTEM EXECUTION SUMMARY")
        
        print("\nTest Results:")
        print(f"  Model Evaluation:    {'PASS' if results['model_evaluation'] else 'FAIL'}")
        print(f"  Victim Scenarios:    See results above")
        
        print("\n" + "="*80)
        print("SAR System Ready - Refactored Architecture Active")
        print("="*80)
        print("\nLog files available at:")
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        print(f"  Main Log:  {os.path.join(logs_dir, 'sar_system.log')}")
        print(f"  Error Log: {os.path.join(logs_dir, 'errors.log')}")
        print("\n" + "="*80)
        
        logger.info("Test scenario execution completed successfully")
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
