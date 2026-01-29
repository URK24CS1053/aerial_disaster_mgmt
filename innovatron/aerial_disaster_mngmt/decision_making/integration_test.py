"""
End-to-end integration test for Fusion Engine + Urgency Scoring
"""

from fusion_engine import fuse_signals, SensorData
from urgency_scoring import assign_urgency
import json


def run_integrated_test(name: str, sensor_data: SensorData, environment_risk: str = "LOW"):
    """Run integrated fusion + urgency test"""
    
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    
    print(f"\nSensor Input:")
    print(json.dumps(sensor_data, indent=2))
    
    # Step 1: Fuse signals
    victim_state = fuse_signals(sensor_data)
    
    print(f"\nFusion Engine Output:")
    print(f"  Presence: {victim_state['presence_confirmed']}")
    print(f"  Responsiveness: {victim_state['responsiveness']}")
    print(f"  Vital Signs: {victim_state['vital_signs']}")
    print(f"  Confidence: {victim_state['confidence_level']}")
    print(f"  Explanations:")
    for exp in victim_state['fusion_explanation']:
        print(f"    - {exp}")
    
    # Step 2: Assign urgency
    urgency = assign_urgency(victim_state, environment_risk)
    
    print(f"\nUrgency Scoring (Environment Risk: {environment_risk}):")
    print(f"  Urgency Level: {urgency['urgency_level']}")
    print(f"  Reasons:")
    for reason in urgency['reason']:
        print(f"    - {reason}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("INTEGRATED FUSION + URGENCY SCORING TESTS")
    print("="*70)
    
    # Test 1: Healthy victim
    run_integrated_test(
        "Healthy Victim - Alert and responsive",
        {
            "human_detected": True,
            "motion_level": "HIGH",
            "heat_presence": "NORMAL",
            "breathing_detected": "YES"
        },
        environment_risk="LOW"
    )
    
    # Test 2: Critically injured
    run_integrated_test(
        "Critical Victim - Unresponsive, weak vitals",
        {
            "human_detected": True,
            "motion_level": "NONE",
            "heat_presence": "LOW",
            "breathing_detected": "NO"
        },
        environment_risk="LOW"
    )
    
    # Test 3: Moderately injured with high environmental risk
    run_integrated_test(
        "Moderate Injury + High Environmental Risk",
        {
            "human_detected": True,
            "motion_level": "LOW",
            "heat_presence": "NORMAL",
            "breathing_detected": "YES"
        },
        environment_risk="HIGH"
    )
    
    # Test 4: Weak signal
    run_integrated_test(
        "Weak Response - Uncertain vitals",
        {
            "human_detected": True,
            "motion_level": "LOW",
            "heat_presence": "NORMAL",
            "breathing_detected": "NO"
        },
        environment_risk="MEDIUM"
    )
    
    # Test 5: False alarm
    run_integrated_test(
        "False Alarm - No human detected",
        {
            "human_detected": False,
            "motion_level": "NONE",
            "heat_presence": "LOW",
            "breathing_detected": "NO"
        },
        environment_risk="LOW"
    )
    
    print("\n" + "="*70)
    print("Integration tests complete!")
    print("="*70)
