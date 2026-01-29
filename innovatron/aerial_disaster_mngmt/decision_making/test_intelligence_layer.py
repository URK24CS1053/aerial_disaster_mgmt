from fusion_engine import fuse_signals
from urgency_scoring import assign_urgency
from datetime import datetime
import json

def run_test_case(name, sensor_data, environment_risk="LOW", location=None):
    print("\n" + "="*50)
    print(f"TEST CASE: {name}")
    print("="*50)

    victim_state = fuse_signals(sensor_data)
    urgency = assign_urgency(victim_state, environment_risk)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    if location:
        print(f"Location: {location}")
    
    print("\nInput Sensor Data:")
    print(json.dumps(sensor_data, indent=2))

    print("\nFused Victim State:")
    print(json.dumps(victim_state, indent=2))

    print("\nUrgency Decision:")
    print(json.dumps(urgency, indent=2))
    print("="*50)


# ================================================
# REAL-WORLD SCENARIO 1: Building Collapse Victim
# ================================================
run_test_case(
    "Building Collapse - Trapped under debris",
    {
        "human_detected": True,
        "motion_level": "MINIMAL",  # Stuck, minimal movement
        "heat_presence": "LOW",      # Body cooling due to exposure
        "breathing_detected": "SHALLOW"  # Weak breathing, possible internal injuries
    },
    environment_risk="HIGH",
    location="Apartment Complex, Floor 3"
)

# ================================================
# REAL-WORLD SCENARIO 2: Earthquake Survivor
# ================================================
run_test_case(
    "Earthquake - Conscious but injured",
    {
        "human_detected": True,
        "motion_level": "LOW",       # Some movement, trying to signal
        "heat_presence": "NORMAL",   # Normal body temperature
        "breathing_detected": "YES"  # Present but possibly painful
    },
    environment_risk="HIGH",
    location="Residential area near epicenter"
)

# ================================================
# REAL-WORLD SCENARIO 3: Flood Victim - Unconscious
# ================================================
run_test_case(
    "Flash Flood - Unconscious in water",
    {
        "human_detected": True,
        "motion_level": "NONE",      # Completely still in water
        "heat_presence": "LOW",      # Hypothermia risk, body temp dropping
        "breathing_detected": "NO"   # No respiratory effort detected
    },
    environment_risk="HIGH",
    location="River valley during flood event"
)

# ================================================
# REAL-WORLD SCENARIO 4: Accident Victim - Conscious
# ================================================
run_test_case(
    "Road Accident - Victim conscious but in shock",
    {
        "human_detected": True,
        "motion_level": "LOW",       # Trying to move but limited
        "heat_presence": "NORMAL",   # Normal temp initially
        "breathing_detected": "YES"  # Rapid/shallow breathing (shock)
    },
    environment_risk="MEDIUM",
    location="Highway, 2km from hospital"
)

# ================================================
# REAL-WORLD SCENARIO 5: Wildfire Evacuation Casualty
# ================================================
run_test_case(
    "Wildfire - Smoke inhalation victim",
    {
        "human_detected": True,
        "motion_level": "WEAK",      # Struggling to move due to smoke
        "heat_presence": "HIGH",     # Extreme heat from fire proximity
        "breathing_detected": "LABORED"  # Breathing difficulty, coughing
    },
    environment_risk="HIGH",
    location="Forest area, 50m from fire front"
)

# ================================================
# REAL-WORLD SCENARIO 6: Heatstroke Victim
# ================================================
run_test_case(
    "Heatwave - Elderly person with heatstroke",
    {
        "human_detected": True,
        "motion_level": "NONE",      # Unconscious/unresponsive
        "heat_presence": "VERY_HIGH",  # Dangerously high body temperature
        "breathing_detected": "IRREGULAR"  # Erratic breathing pattern
    },
    environment_risk="MEDIUM",
    location="Urban area during extreme heat warning"
)

# ================================================
# REAL-WORLD SCENARIO 7: Avalanche Burial
# ================================================
run_test_case(
    "Avalanche - Person buried under snow",
    {
        "human_detected": True,
        "motion_level": "NONE",      # Fully buried, cannot move
        "heat_presence": "LOW",      # Buried under insulating snow
        "breathing_detected": "BARELY_DETECTABLE"  # Minimal oxygen
    },
    environment_risk="HIGH",
    location="Mountain slope at 3500m elevation"
)

# ================================================
# REAL-WORLD SCENARIO 8: False Alarm
# ================================================
run_test_case(
    "False Detection - Animal mistaken for human",
    {
        "human_detected": False,     # Actually an animal
        "motion_level": "NONE",
        "heat_presence": "LOW",
        "breathing_detected": "NO"
    },
    environment_risk="LOW",
    location="Forest search area"
)

# ================================================
# REAL-WORLD SCENARIO 9: Responsive Survivor
# ================================================
run_test_case(
    "Disaster Survivor - Alert and responsive",
    {
        "human_detected": True,
        "motion_level": "HIGH",      # Moving around, communicating
        "heat_presence": "NORMAL",   # Normal vitals
        "breathing_detected": "YES"  # Normal breathing
    },
    environment_risk="LOW",
    location="Safe zone at evacuation center"
)
9