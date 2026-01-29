"""
Test and demonstrate the logging system
"""

from fusion_engine import fuse_signals
from urgency_scoring import assign_urgency
from logger_config import get_logger
import os

logger = get_logger("logging_test")


def test_logging():
    """Demonstrate logging in action"""
    
    print("\n" + "="*70)
    print("LOGGING SYSTEM TEST")
    print("="*70)
    
    logger.info("Starting logging system test")
    
    # Test 1: Valid scenario with logging
    print("\nTest 1: Valid victim scenario (logging will show details)")
    logger.info("Test 1: Processing healthy victim")
    
    sensor_data = {
        "human_detected": True,
        "motion_level": "HIGH",
        "heat_presence": "NORMAL",
        "breathing_detected": "YES"
    }
    
    try:
        victim_state = fuse_signals(sensor_data)
        urgency = assign_urgency(victim_state, "LOW")
        print(f"  Confidence: {victim_state['confidence_level']}")
        print(f"  Urgency: {urgency['urgency_level']}")
        logger.info(f"Test 1 completed successfully")
    except Exception as e:
        logger.error(f"Test 1 failed: {str(e)}", exc_info=True)
    
    # Test 2: Critical scenario
    print("\nTest 2: Critical victim scenario")
    logger.info("Test 2: Processing critical victim")
    
    sensor_data = {
        "human_detected": True,
        "motion_level": "NONE",
        "heat_presence": "LOW",
        "breathing_detected": "NO"
    }
    
    try:
        victim_state = fuse_signals(sensor_data)
        urgency = assign_urgency(victim_state, "HIGH")
        print(f"  Confidence: {victim_state['confidence_level']}")
        print(f"  Urgency: {urgency['urgency_level']}")
        logger.info(f"Test 2 completed successfully")
    except Exception as e:
        logger.error(f"Test 2 failed: {str(e)}", exc_info=True)
    
    # Test 3: Invalid data (will log error)
    print("\nTest 3: Invalid sensor data (error logging)")
    logger.info("Test 3: Processing invalid sensor data")
    
    invalid_data = {
        "human_detected": True,
        "motion_level": "INVALID",
        "heat_presence": "NORMAL",
        "breathing_detected": "YES"
    }
    
    try:
        victim_state = fuse_signals(invalid_data)
        logger.info("Test 3 completed")
    except ValueError as e:
        print(f"  Error caught: {str(e)}")
        logger.info(f"Test 3 correctly raised ValueError")
    except Exception as e:
        logger.error(f"Test 3 unexpected error: {str(e)}", exc_info=True)
    
    # Test 4: False alarm
    print("\nTest 4: False alarm scenario")
    logger.info("Test 4: Processing false alarm (no human detected)")
    
    sensor_data = {
        "human_detected": False,
        "motion_level": "NONE",
        "heat_presence": "LOW",
        "breathing_detected": "NO"
    }
    
    try:
        victim_state = fuse_signals(sensor_data)
        urgency = assign_urgency(victim_state, "LOW")
        print(f"  Urgency: {urgency['urgency_level']}")
        logger.info(f"Test 4 completed successfully")
    except Exception as e:
        logger.error(f"Test 4 failed: {str(e)}", exc_info=True)
    
    print("\n" + "="*70)
    print("LOGGING SYSTEM TEST COMPLETE")
    print("="*70)
    
    # Display log file locations
    print("\nLog File Locations:")
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    main_log = os.path.join(logs_dir, "sar_system.log")
    error_log = os.path.join(logs_dir, "errors.log")
    
    print(f"  Main Log: {main_log}")
    print(f"  Error Log: {error_log}")
    
    if os.path.exists(main_log):
        print(f"\nMain Log File (last 20 lines):")
        print("-" * 70)
        with open(main_log, 'r') as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_logging()
