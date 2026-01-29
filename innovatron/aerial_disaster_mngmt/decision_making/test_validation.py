"""
Test input validation and error handling
"""

from fusion_engine import fuse_signals, validate_sensor_data


def test_validation():
    """Test input validation"""
    
    print("="*70)
    print("INPUT VALIDATION TESTS")
    print("="*70)
    
    # Test 1: Valid data
    print("\nTest 1: Valid sensor data")
    valid_data = {
        "human_detected": True,
        "motion_level": "HIGH",
        "heat_presence": "NORMAL",
        "breathing_detected": "YES"
    }
    is_valid, msg = validate_sensor_data(valid_data)
    print(f"  Result: {'PASS' if is_valid else 'FAIL'} - {msg if not is_valid else 'Valid'}")
    
    # Test 2: Missing field
    print("\nTest 2: Missing required field")
    invalid_data = {
        "human_detected": True,
        "motion_level": "HIGH",
        "heat_presence": "NORMAL"
        # missing breathing_detected
    }
    is_valid, msg = validate_sensor_data(invalid_data)
    print(f"  Result: {'✓ PASS' if not is_valid else '✗ FAIL'}")
    print(f"  Error: {msg}")
    
    # Test 3: Invalid motion_level
    print("\nTest 3: Invalid motion_level")
    invalid_data = {
        "human_detected": True,
        "motion_level": "EXTREME",  # Invalid
        "heat_presence": "NORMAL",
        "breathing_detected": "YES"
    }
    is_valid, msg = validate_sensor_data(invalid_data)
    print(f"  Result: {'PASS' if not is_valid else 'FAIL'}")
    print(f"  Error: {msg}")
    
    # Test 4: Invalid heat_presence
    print("\nTest 4: Invalid heat_presence")
    invalid_data = {
        "human_detected": True,
        "motion_level": "HIGH",
        "heat_presence": "EXTREME",  # Invalid
        "breathing_detected": "YES"
    }
    is_valid, msg = validate_sensor_data(invalid_data)
    print(f"  Result: {'PASS' if not is_valid else 'FAIL'}")
    print(f"  Error: {msg}")
    
    # Test 5: Invalid breathing_detected
    print("\nTest 5: Invalid breathing_detected")
    invalid_data = {
        "human_detected": True,
        "motion_level": "HIGH",
        "heat_presence": "NORMAL",
        "breathing_detected": "MAYBE"  # Invalid
    }
    is_valid, msg = validate_sensor_data(invalid_data)
    print(f"  Result: {'PASS' if not is_valid else 'FAIL'}")
    print(f"  Error: {msg}")
    
    # Test 6: Wrong type for human_detected
    print("\nTest 6: Wrong type for human_detected")
    invalid_data = {
        "human_detected": "YES",  # Should be bool
        "motion_level": "HIGH",
        "heat_presence": "NORMAL",
        "breathing_detected": "YES"
    }
    is_valid, msg = validate_sensor_data(invalid_data)
    print(f"  Result: {'PASS' if not is_valid else 'FAIL'}")
    print(f"  Error: {msg}")
    
    print("\n" + "="*70)


def test_error_handling():
    """Test error handling in fusion function"""
    
    print("\n" + "="*70)
    print("ERROR HANDLING TESTS")
    print("="*70)
    
    # Test 1: Invalid data raises exception
    print("\nTest 1: Invalid data raises ValueError")
    invalid_data = {
        "human_detected": True,
        "motion_level": "INVALID",
        "heat_presence": "NORMAL",
        "breathing_detected": "YES"
    }
    
    try:
        fuse_signals(invalid_data)
        print("  Result: FAIL - No exception raised")
    except ValueError as e:
        print(f"  Result: PASS - Exception raised")
        print(f"  Error: {str(e)}")
    
    # Test 2: Valid data processes without error
    print("\nTest 2: Valid data processes without error")
    valid_data = {
        "human_detected": True,
        "motion_level": "HIGH",
        "heat_presence": "NORMAL",
        "breathing_detected": "YES"
    }
    
    try:
        result = fuse_signals(valid_data)
        print(f"  Result: PASS - Processed successfully")
        print(f"  Confidence: {result['confidence_level']}")
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_validation()
    test_error_handling()
    print("\nAll validation and error handling tests complete!")
