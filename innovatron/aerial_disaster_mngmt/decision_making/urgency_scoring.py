# urgency_scoring.py
from typing import Dict, List, Literal, TypedDict
from fusion_engine import VictimState
import json
from logger_config import get_logger

logger = get_logger("urgency_scoring")


UrgencyLevel = Literal["CRITICAL", "HIGH", "MODERATE", "NONE"]
EnvironmentRisk = Literal["LOW", "MEDIUM", "HIGH"]


class UrgencyResult(TypedDict):
    urgency_level: UrgencyLevel
    reason: List[str]


def assign_urgency(
    victim_state: VictimState,
    environment_risk: EnvironmentRisk = "LOW"
) -> UrgencyResult:
    """
    Assign urgency level based on victim state and environmental conditions.
    
    Args:
        victim_state: The fused victim state from the sensor fusion engine
        environment_risk: Environmental risk level (LOW, MEDIUM, HIGH)
    
    Returns:
        UrgencyResult with urgency level and reasoning
    """

    if not victim_state["presence_confirmed"]:
        logger.info("Assigning NONE urgency - no human presence confirmed")
        return {
            "urgency_level": "NONE",
            "reason": ["No confirmed human presence"]
        }

    urgency: UrgencyLevel = "MODERATE"
    explanation: List[str] = []

    # Extract confidence level for additional decision making
    confidence = victim_state["confidence_level"]
    responsiveness = victim_state["responsiveness"]
    vital_signs = victim_state["vital_signs"]

    logger.debug(f"Assessing urgency - Responsiveness: {responsiveness}, Vitals: {vital_signs}, Confidence: {confidence}")

    # Critical conditions - requires immediate action
    if (
        responsiveness == "UNRESPONSIVE"
        and vital_signs == "WEAK_SIGNS"
    ):
        urgency = "CRITICAL"
        explanation.append(
            "CRITICAL: Victim unresponsive with weak vital indicators - immediate rescue needed"
        )
        logger.warning(f"CRITICAL urgency assigned: Unresponsive with weak vitals")
    
    # Critical due to high confidence in bad state
    elif confidence == "HIGH" and responsiveness == "UNRESPONSIVE":
        urgency = "CRITICAL"
        explanation.append(
            "CRITICAL: High confidence victim is unresponsive - immediate intervention required"
        )
        logger.warning(f"CRITICAL urgency assigned: High confidence unresponsive")

    # High urgency conditions - urgent response needed
    elif (
        responsiveness in ["UNRESPONSIVE", "WEAK_RESPONSE"]
        and vital_signs in ["UNCERTAIN_SIGNS", "WEAK_SIGNS"]
    ):
        urgency = "HIGH"
        explanation.append(
            "HIGH: Victim shows limited responsiveness and unstable indicators"
        )
        logger.info(f"HIGH urgency assigned: Limited responsiveness + unstable vitals")
    
    # High urgency if weak response with weak signals
    elif responsiveness == "WEAK_RESPONSE" and vital_signs == "WEAK_SIGNS":
        urgency = "HIGH"
        explanation.append(
            "HIGH: Weak responsiveness combined with weak vital signs"
        )
        logger.info(f"HIGH urgency assigned: Weak response + weak signals")

    # Moderate urgency - standard response
    else:
        urgency = "MODERATE"
        explanation.append(
            "MODERATE: Victim shows responsiveness or stable indicators"
        )
        logger.info(f"MODERATE urgency assigned: Responsive or stable state")

    # Environmental risk escalation
    if environment_risk == "HIGH":
        logger.warning(f"HIGH environmental risk detected - escalating urgency")
        if urgency == "MODERATE":
            urgency = "HIGH"
            explanation.append(
                "Escalated to HIGH due to high environmental risk"
            )
        elif urgency == "HIGH":
            urgency = "CRITICAL"
            explanation.append(
                "Escalated to CRITICAL due to high environmental risk"
            )
    
    elif environment_risk == "MEDIUM":
        logger.info(f"MEDIUM environmental risk detected")
        if urgency == "MODERATE" and confidence == "HIGH":
            urgency = "HIGH"
            explanation.append(
                "Escalated to HIGH due to medium environmental risk and high confidence"
            )

    logger.info(f"Final urgency level assigned: {urgency}")
    
    return {
        "urgency_level": urgency,
        "reason": explanation
    }


# ---- Test Cases ----

def run_urgency_test(name: str, victim_state: VictimState, environment_risk: EnvironmentRisk = "LOW"):
    """Run a single test case for urgency scoring"""
    print(f"\nTest: {name}")
    print("-" * 70)
    print(f"  Victim State:")
    print(f"    - Presence: {victim_state['presence_confirmed']}")
    print(f"    - Responsiveness: {victim_state['responsiveness']}")
    print(f"    - Vital Signs: {victim_state['vital_signs']}")
    print(f"    - Confidence: {victim_state['confidence_level']}")
    print(f"  Environment Risk: {environment_risk}")
    
    result = assign_urgency(victim_state, environment_risk)
    
    print(f"\n  Result:")
    print(f"    - Urgency Level: {result['urgency_level']}")
    print(f"    - Reasons:")
    for reason in result['reason']:
        print(f"      * {reason}")


def run_all_tests():
    """Run comprehensive urgency scoring tests"""
    
    print("\n" + "=" * 70)
    print("URGENCY SCORING TEST CASES")
    print("=" * 70)
    
    # Test 1: Critical case - unresponsive with weak vitals
    run_urgency_test(
        "CRITICAL: Unresponsive with weak vitals",
        {
            "presence_confirmed": True,
            "responsiveness": "UNRESPONSIVE",
            "vital_signs": "WEAK_SIGNS",
            "confidence_level": "HIGH",
            "fusion_explanation": []
        },
        environment_risk="LOW"
    )
    
    # Test 2: Critical with environmental escalation
    run_urgency_test(
        "CRITICAL: High environment risk + HIGH confidence unresponsive",
        {
            "presence_confirmed": True,
            "responsiveness": "UNRESPONSIVE",
            "vital_signs": "UNCERTAIN_SIGNS",
            "confidence_level": "HIGH",
            "fusion_explanation": []
        },
        environment_risk="HIGH"
    )
    
    # Test 3: High urgency
    run_urgency_test(
        "HIGH: Weak response with weak signals",
        {
            "presence_confirmed": True,
            "responsiveness": "WEAK_RESPONSE",
            "vital_signs": "WEAK_SIGNS",
            "confidence_level": "MEDIUM",
            "fusion_explanation": []
        },
        environment_risk="LOW"
    )
    
    # Test 4: Moderate - responsive
    run_urgency_test(
        "MODERATE: Responsive with stable vitals",
        {
            "presence_confirmed": True,
            "responsiveness": "RESPONSIVE",
            "vital_signs": "STABLE_SIGNS",
            "confidence_level": "HIGH",
            "fusion_explanation": []
        },
        environment_risk="LOW"
    )
    
    # Test 5: Moderate escalated to HIGH
    run_urgency_test(
        "HIGH: Moderate with HIGH environmental risk",
        {
            "presence_confirmed": True,
            "responsiveness": "RESPONSIVE",
            "vital_signs": "UNCERTAIN_SIGNS",
            "confidence_level": "MEDIUM",
            "fusion_explanation": []
        },
        environment_risk="HIGH"
    )
    
    # Test 6: No detection
    run_urgency_test(
        "NONE: No human presence detected",
        {
            "presence_confirmed": False,
            "responsiveness": "UNKNOWN",
            "vital_signs": "UNKNOWN",
            "confidence_level": "LOW",
            "fusion_explanation": ["No human detected"]
        },
        environment_risk="LOW"
    )
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_tests()
