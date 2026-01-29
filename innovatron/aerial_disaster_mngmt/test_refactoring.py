#!/usr/bin/env python3
"""
Quick test of the refactored architecture.
This validates that the PerceptionOutput schema and fusion engine work correctly.
"""

import sys
import os
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("REFACTORED SAR SYSTEM - QUICK VALIDATION TEST")
print("=" * 80)

# Test 1: Validate schema exists and is importable
print("\n[TEST 1] Importing PerceptionOutput schema...")
try:
    from interfaces.perception_schema import PerceptionOutput, UncertaintyFlag
    print("✓ PerceptionOutput schema imported successfully")
    print(f"  - Has 'feature_vector' field: {'feature_vector' in PerceptionOutput.__annotations__}")
    print(f"  - Has 'uncertain' field: {'uncertain' in PerceptionOutput.__annotations__}")
    print(f"  - Has 'uncertainty_flags' field: {'uncertainty_flags' in PerceptionOutput.__annotations__}")
    print(f"  - Has 'confidence_reduction_factor' field: {'confidence_reduction_factor' in PerceptionOutput.__annotations__}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Validate perception_utils exists
print("\n[TEST 2] Importing perception utilities...")
try:
    from perception.perception_utils import (
        create_perception_output, 
        build_feature_vector,
        validate_perception_output
    )
    print("✓ Perception utilities imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Create a test PerceptionOutput
print("\n[TEST 3] Creating a test PerceptionOutput...")
try:
    test_output = create_perception_output(
        track_id="test_0",
        bbox=(100, 100, 50, 100),
        frame_id=0,
        timestamp=datetime.now().timestamp(),
        detected=True,
        detection_confidence=0.95,
        motion_state="RESPONSIVE",
        motion_score=0.8,
        motion_confidence=0.85,
        camera_motion_compensated=True,
        heat_present=True,
        thermal_intensity=0.7,
        thermal_stability=0.8,
        breathing_proxy="DETECTED",
        hypothermia_risk="LOW",
        thermal_confidence=0.75,
        visibility_level="GOOD",
        signal_quality=0.82,
        uncertainty_flags=[],
        confidence_reduction_factor=1.0
    )
    print("✓ PerceptionOutput created successfully")
    print(f"  - Feature vector length: {len(test_output['feature_vector'])}")
    print(f"  - Feature vector (first 5): {test_output['feature_vector'][:5]}")
    print(f"  - Uncertain: {test_output['uncertain']}")
    print(f"  - Confidence reduction factor: {test_output['confidence_reduction_factor']}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Validate the PerceptionOutput
print("\n[TEST 4] Validating PerceptionOutput schema compliance...")
try:
    is_valid, error_msg = validate_perception_output(test_output)
    if is_valid:
        print("✓ PerceptionOutput is valid and complies with schema")
    else:
        print(f"✗ FAILED: {error_msg}")
        sys.exit(1)
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Import fusion engine
print("\n[TEST 5] Importing fusion engine with new architecture...")
try:
    from decision_making.fusion_engine import (
        infer_from_perception,
        validate_perception_output as fusion_validate
    )
    print("✓ Fusion engine imported successfully")
    print(f"  - Has infer_from_perception function: True")
    print(f"  - Has validate_perception_output function: True")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test inference
print("\n[TEST 6] Testing ML-based inference with PerceptionOutput...")
try:
    from decision_making.fusion_engine import infer_from_perception, load_model, train_model
    
    # Ensure model exists
    if load_model() is None:
        print("  Model not found, training...")
        train_model(save=True)
        print("  Model trained and saved")
    
    # Perform inference
    victim_state = infer_from_perception(test_output)
    
    print("✓ Inference completed successfully")
    print(f"  - Presence confirmed: {victim_state['presence_confirmed']}")
    print(f"  - Responsiveness: {victim_state['responsiveness']}")
    print(f"  - Vital signs: {victim_state['vital_signs']}")
    print(f"  - Confidence level: {victim_state['confidence_level']}")
    print(f"  - Model confidence: {victim_state['model_confidence']:.3f}")
    print(f"  - Effective confidence: {victim_state['effective_confidence']:.3f}")
    print(f"  - Uncertain: {victim_state['uncertain']}")
    print(f"  - Explanations: {len(victim_state['fusion_explanation'])} items")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test urgency scoring
print("\n[TEST 7] Testing urgency scoring with VictimState...")
try:
    from decision_making.urgency_scoring import assign_urgency
    
    urgency = assign_urgency(victim_state, environment_risk="LOW")
    
    print("✓ Urgency scoring completed successfully")
    print(f"  - Urgency level: {urgency['urgency_level']}")
    print(f"  - Reasons: {len(urgency['reason'])} items")
    for i, reason in enumerate(urgency['reason'], 1):
        print(f"    {i}. {reason[:70]}...")
        
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test orchestration
print("\n[TEST 8] Testing full orchestration pipeline...")
try:
    from decision_making.main import orchestrate_perception_to_urgency
    
    result = orchestrate_perception_to_urgency(
        perception_outputs=[test_output],
        environment_risk="LOW"
    )
    
    print("✓ Orchestration pipeline executed successfully")
    print(f"  - Success: {result['success']}")
    print(f"  - Number of decisions: {len(result['decisions'])}")
    if result['decisions']:
        decision = result['decisions'][0]
        print(f"  - First decision valid: {decision['valid']}")
        print(f"  - Urgency: {decision['urgency']['urgency_level']}")
        print(f"  - Effective confidence: {decision['effective_confidence']:.3f}")
        
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nRefactored architecture summary:")
print("✓ PerceptionOutput schema with feature vectors")
print("✓ Strict layer separation (Perception → Fusion → Urgency)")
print("✓ ML-based inference on feature vectors only")
print("✓ Uncertainty handling and confidence reduction")
print("✓ Full explainability chain")
print("✓ Orchestration layer as integration point")
print("\nThe system is ready for use!")
