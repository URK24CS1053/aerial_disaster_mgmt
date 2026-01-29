# perception/perception_utils.py
"""
Utility functions for building PerceptionOutput objects with proper feature vectors
and uncertainty handling. Ensures strict schema compliance across all perception modules.
"""

from typing import Dict, List, Optional
import sys
from datetime import datetime

# Import schema
sys.path.insert(0, '..')
from interfaces.perception_schema import (
    PerceptionOutput, MotionState, BreathingProxy, VisibilityLevel, 
    UncertaintyFlag
)


def motion_state_to_numeric(motion_state: MotionState) -> int:
    """Convert MotionState enum to numeric feature value."""
    mapping = {
        "NO_RESPONSE": 0,
        "LIMITED_RESPONSE": 1,
        "RESPONSIVE": 2,
        "UNCERTAIN": 3
    }
    return mapping.get(motion_state, 3)


def breathing_proxy_to_numeric(breathing: BreathingProxy) -> int:
    """Convert BreathingProxy enum to numeric feature value."""
    mapping = {
        "NOT_DETECTED": 0,
        "DETECTED": 1,
        "UNCERTAIN": 2
    }
    return mapping.get(breathing, 2)


def visibility_to_numeric(visibility: VisibilityLevel) -> int:
    """Convert VisibilityLevel enum to numeric feature value."""
    mapping = {
        "UNKNOWN": 0,
        "POOR": 1,
        "GOOD": 2
    }
    return mapping.get(visibility, 0)


def build_feature_vector(
    detected: bool,
    detection_confidence: float,
    motion_state: MotionState,
    motion_score: float,
    motion_confidence: float,
    heat_present: bool,
    thermal_intensity: float,
    thermal_stability: float,
    breathing_proxy: BreathingProxy,
    breathing_confidence: float,
    thermal_confidence: float,
    visibility_level: VisibilityLevel,
    signal_quality: float,
    uncertainty_flags: List[UncertaintyFlag]
) -> List[float]:
    """
    Build a 14-element feature vector for ML inference.
    
    Feature indices:
        [0]  - Human detected (0 or 1)
        [1]  - Detection confidence (0.0-1.0)
        [2]  - Motion state (0-3)
        [3]  - Motion score (0.0-1.0)
        [4]  - Motion confidence (0.0-1.0)
        [5]  - Heat present (0 or 1)
        [6]  - Thermal intensity (0.0-1.0)
        [7]  - Thermal stability (0.0-1.0)
        [8]  - Breathing proxy (0-2)
        [9]  - Breathing confidence (0.0-1.0)
        [10] - Thermal confidence (0.0-1.0)
        [11] - Visibility level (0-2)
        [12] - Signal quality (0.0-1.0)
        [13] - Number of uncertainty flags (0-6)
    
    Returns:
        14-element list of floats, ML-ready and normalized
    """
    return [
        float(1 if detected else 0),           # [0] detected (binary)
        float(max(0.0, min(1.0, detection_confidence))),  # [1] detection_confidence
        float(motion_state_to_numeric(motion_state)),     # [2] motion_state
        float(max(0.0, min(1.0, motion_score))),          # [3] motion_score
        float(max(0.0, min(1.0, motion_confidence))),     # [4] motion_confidence
        float(1 if heat_present else 0),       # [5] heat_present (binary)
        float(max(0.0, min(1.0, thermal_intensity))),     # [6] thermal_intensity
        float(max(0.0, min(1.0, thermal_stability))),     # [7] thermal_stability
        float(breathing_proxy_to_numeric(breathing_proxy)),# [8] breathing_proxy
        float(max(0.0, min(1.0, breathing_confidence))),  # [9] breathing_confidence
        float(max(0.0, min(1.0, thermal_confidence))),    # [10] thermal_confidence
        float(visibility_to_numeric(visibility_level)),   # [11] visibility_level
        float(max(0.0, min(1.0, signal_quality))),        # [12] signal_quality
        float(len(uncertainty_flags))          # [13] num_uncertainty_flags
    ]


def create_perception_output(
    track_id: str,
    bbox: tuple,
    frame_id: int,
    timestamp: float,
    detected: bool,
    detection_confidence: float,
    motion_state: MotionState,
    motion_score: float,
    motion_confidence: float,
    camera_motion_compensated: bool,
    heat_present: bool,
    thermal_intensity: float,
    thermal_stability: float,
    breathing_proxy: BreathingProxy,
    hypothermia_risk: str,
    thermal_confidence: float,
    visibility_level: VisibilityLevel,
    signal_quality: float,
    uncertainty_flags: Optional[List[UncertaintyFlag]] = None,
    confidence_reduction_factor: float = 1.0,
    debug_info: Optional[dict] = None
) -> PerceptionOutput:
    """
    Factory function to create a valid PerceptionOutput object.
    
    Args:
        track_id: Unique identifier for this detection
        bbox: (x, y, w, h) bounding box
        frame_id: Frame number
        timestamp: Unix timestamp
        detected: Whether human detected
        detection_confidence: 0-1
        motion_state: MotionState enum
        motion_score: 0-1 normalized
        motion_confidence: 0-1
        camera_motion_compensated: Whether camera motion was filtered
        heat_present: Whether heat signature detected
        thermal_intensity: 0-1 normalized
        thermal_stability: 0-1 temporal consistency
        breathing_proxy: BreathingProxy enum
        hypothermia_risk: "LOW" or "HIGH"
        thermal_confidence: 0-1
        visibility_level: VisibilityLevel enum
        signal_quality: 0-1 aggregate reliability
        uncertainty_flags: List of UncertaintyFlag values
        confidence_reduction_factor: 0-1, how much to reduce confidence
        debug_info: Optional debug metadata
    
    Returns:
        PerceptionOutput dict conforming to schema
    """
    if uncertainty_flags is None:
        uncertainty_flags = []
    
    # Clamp all confidence values to [0, 1]
    detection_confidence = max(0.0, min(1.0, detection_confidence))
    motion_score = max(0.0, min(1.0, motion_score))
    motion_confidence = max(0.0, min(1.0, motion_confidence))
    thermal_intensity = max(0.0, min(1.0, thermal_intensity))
    thermal_stability = max(0.0, min(1.0, thermal_stability))
    thermal_confidence = max(0.0, min(1.0, thermal_confidence))
    signal_quality = max(0.0, min(1.0, signal_quality))
    confidence_reduction_factor = max(0.0, min(1.0, confidence_reduction_factor))
    
    # Build feature vector
    feature_vector = build_feature_vector(
        detected=detected,
        detection_confidence=detection_confidence,
        motion_state=motion_state,
        motion_score=motion_score,
        motion_confidence=motion_confidence,
        heat_present=heat_present,
        thermal_intensity=thermal_intensity,
        thermal_stability=thermal_stability,
        breathing_proxy=breathing_proxy,
        breathing_confidence=detection_confidence,  # Use breathing from breathing_proxy
        thermal_confidence=thermal_confidence,
        visibility_level=visibility_level,
        signal_quality=signal_quality,
        uncertainty_flags=uncertainty_flags
    )
    
    # Create output dict
    output: PerceptionOutput = {
        "track_id": track_id,
        "bbox": bbox,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "detected": detected,
        "detection_confidence": detection_confidence,
        "motion_state": motion_state,
        "motion_score": motion_score,
        "motion_confidence": motion_confidence,
        "camera_motion_compensated": camera_motion_compensated,
        "heat_present": heat_present,
        "thermal_intensity": thermal_intensity,
        "thermal_stability": thermal_stability,
        "breathing_proxy": breathing_proxy,
        "hypothermia_risk": hypothermia_risk,
        "thermal_confidence": thermal_confidence,
        "visibility_level": visibility_level,
        "signal_quality": signal_quality,
        "feature_vector": feature_vector,
        "uncertain": len(uncertainty_flags) > 0,
        "uncertainty_flags": uncertainty_flags,
        "confidence_reduction_factor": confidence_reduction_factor,
        "debug_info": debug_info
    }
    
    return output


def apply_confidence_reduction(
    perception_output: PerceptionOutput,
    reason: str,
    reduction_factor: float
) -> PerceptionOutput:
    """
    Apply confidence reduction to a PerceptionOutput (e.g., due to global motion or poor visibility).
    
    Args:
        perception_output: The perception output to modify
        reason: UncertaintyFlag indicating the reason for reduction
        reduction_factor: How much to reduce (0 = complete loss, 1 = no reduction)
    
    Returns:
        Modified PerceptionOutput with reduced confidence values
    """
    # Add uncertainty flag if not already present
    if reason not in perception_output["uncertainty_flags"]:
        perception_output["uncertainty_flags"].append(reason)
    
    # Update uncertainty flag
    perception_output["uncertain"] = len(perception_output["uncertainty_flags"]) > 0
    
    # Apply reduction factor to all confidence values
    current_reduction = perception_output["confidence_reduction_factor"]
    perception_output["confidence_reduction_factor"] = current_reduction * reduction_factor
    
    # Rebuild feature vector with updated uncertainties
    perception_output["feature_vector"] = build_feature_vector(
        detected=perception_output["detected"],
        detection_confidence=perception_output["detection_confidence"] * reduction_factor,
        motion_state=perception_output["motion_state"],
        motion_score=perception_output["motion_score"],
        motion_confidence=perception_output["motion_confidence"] * reduction_factor,
        heat_present=perception_output["heat_present"],
        thermal_intensity=perception_output["thermal_intensity"],
        thermal_stability=perception_output["thermal_stability"],
        breathing_proxy=perception_output["breathing_proxy"],
        breathing_confidence=perception_output["detection_confidence"] * reduction_factor,
        thermal_confidence=perception_output["thermal_confidence"] * reduction_factor,
        visibility_level=perception_output["visibility_level"],
        signal_quality=perception_output["signal_quality"] * reduction_factor,
        uncertainty_flags=perception_output["uncertainty_flags"]
    )
    
    return perception_output


def validate_perception_output(output: dict) -> tuple[bool, str]:
    """
    Validate that a dict conforms to PerceptionOutput schema.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = {
        "track_id", "bbox", "frame_id", "timestamp",
        "detected", "detection_confidence",
        "motion_state", "motion_score", "motion_confidence", "camera_motion_compensated",
        "heat_present", "thermal_intensity", "thermal_stability", "breathing_proxy",
        "hypothermia_risk", "thermal_confidence",
        "visibility_level", "signal_quality",
        "feature_vector", "uncertain", "uncertainty_flags", "confidence_reduction_factor",
        "debug_info"
    }
    
    # Check required fields
    missing = required_fields - set(output.keys())
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Check feature vector length
    if not isinstance(output["feature_vector"], list) or len(output["feature_vector"]) != 14:
        return False, f"feature_vector must be a list of 14 floats, got {len(output.get('feature_vector', []))}"
    
    # Check confidence values are in [0, 1]
    confidence_fields = [
        "detection_confidence", "motion_confidence", "thermal_confidence", "signal_quality"
    ]
    for field in confidence_fields:
        value = output.get(field, 0)
        if not (0.0 <= value <= 1.0):
            return False, f"{field} must be in [0, 1], got {value}"
    
    # Check enums
    valid_motion_states = {"RESPONSIVE", "LIMITED_RESPONSE", "NO_RESPONSE", "UNCERTAIN"}
    if output["motion_state"] not in valid_motion_states:
        return False, f"motion_state must be one of {valid_motion_states}, got {output['motion_state']}"
    
    valid_breathing = {"DETECTED", "NOT_DETECTED", "UNCERTAIN"}
    if output["breathing_proxy"] not in valid_breathing:
        return False, f"breathing_proxy must be one of {valid_breathing}, got {output['breathing_proxy']}"
    
    valid_visibility = {"GOOD", "POOR", "UNKNOWN"}
    if output["visibility_level"] not in valid_visibility:
        return False, f"visibility_level must be one of {valid_visibility}, got {output['visibility_level']}"
    
    return True, ""
