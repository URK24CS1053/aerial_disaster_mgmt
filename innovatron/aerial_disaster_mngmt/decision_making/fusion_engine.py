# decision_making/fusion_engine.py
"""
Fusion Engine: ML-Based Inference Layer

Consumes PerceptionOutput exclusively (from perception layer).
Performs multi-modal fusion using feature_vector from perception outputs.
Does NOT access raw sensor data or perception internals.

Architecture:
- Input: PerceptionOutput(s) with feature vectors
- Processing: ML-based probabilistic inference
- Output: VictimState with confidence and explainability
- Uncertainty: Handles UNCERTAIN states gracefully via confidence reduction factors
"""

from typing import Dict, List, Literal, TypedDict, Optional
import pickle
import os
import csv
import sys
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Import schema and utilities
# Add parent directory to path for imports (works from decision_making/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces.perception_schema import PerceptionOutput
from logger_config import get_logger

logger = get_logger("fusion_engine")


# ---- Type Definitions ----

ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]
Responsiveness = Literal["RESPONSIVE", "WEAK_RESPONSE", "UNRESPONSIVE", "UNCERTAIN"]
VitalSigns = Literal["STABLE_SIGNS", "UNCERTAIN_SIGNS", "WEAK_SIGNS", "UNCERTAIN"]


class VictimState(TypedDict):
    """Output of fusion engine - explainable victim assessment."""
    presence_confirmed: bool
    responsiveness: Responsiveness
    vital_signs: VitalSigns
    confidence_level: ConfidenceLevel
    model_confidence: float          # 0-1, model's output probability
    effective_confidence: float      # 0-1, after applying reduction factors
    fusion_explanation: List[str]
    uncertain: bool                  # Whether state is uncertain


# ---- Model Management ----

_model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fusion_model.pkl")


def _create_training_data():
    """Load training data from CSV dataset.
    
    Legacy function for backward compatibility. Converts old SensorData format to feature vectors.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "sensor_training_data.csv")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    X = []
    y = []
    
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Old format: 4 features (human_detected, motion_level, heat_presence, breathing_detected)
            # Map to new format by padding with zeros to 14 features
            features = [
                int(row['human_detected']),           # [0] detected
                0.5,                                  # [1] detection_confidence (assume 0.5)
                int(row['motion_level']),             # [2] motion_state (NONE=0, LOW=1, HIGH=2)
                int(row['motion_level']) / 2.0,       # [3] motion_score (normalized)
                0.5,                                  # [4] motion_confidence (assume 0.5)
                int(row['heat_presence']),            # [5] heat_present
                int(row['heat_presence']) / 2.0,      # [6] thermal_intensity (normalized)
                0.5,                                  # [7] thermal_stability (assume 0.5)
                int(row['breathing_detected'] == "YES"),  # [8] breathing_proxy
                0.5,                                  # [9] breathing_confidence (assume 0.5)
                0.5,                                  # [10] thermal_confidence (assume 0.5)
                2,                                    # [11] visibility_level (GOOD=2)
                0.6,                                  # [12] signal_quality (assume 0.6)
                0                                     # [13] num_uncertainty_flags
            ]
            label = int(row['confidence_level'])
            X.append(features)
            y.append(label)
    
    logger.info(f"Loaded {len(X)} training samples from sensor_training_data.csv")
    return np.array(X), np.array(y)


def save_model():
    """Save trained model to disk."""
    global _model
    if _model is None:
        logger.error("Attempted to save model but no model is trained")
        raise ValueError("No trained model to save")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(_model, f)
    logger.info(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load model from disk if it exists."""
    global _model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        logger.info(f"Model loaded from {MODEL_PATH}")
        return _model
    logger.debug("No saved model found, will train new model")
    return None


def train_model(save=True):
    """Train the fusion model and optionally save it."""
    global _model
    logger.info("Starting model training...")
    X, y = _create_training_data()
    _model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    _model.fit(X, y)
    logger.info(f"Model trained with {len(X)} samples")
    
    if save:
        save_model()
    
    return _model


def evaluate_model_performance(X, y):
    """Evaluate model performance using cross-validation and metrics.
    
    Returns:
        Dictionary with evaluation metrics
    """
    global _model
    
    if _model is None:
        return {"error": "Model not trained"}
    
    # Cross-validation scores
    cv_scores = cross_val_score(_model, X, y, cv=5, scoring='accuracy')
    
    # Predictions for detailed metrics
    y_pred = _model.predict(X)
    
    # Calculate metrics
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    
    return {
        "cross_validation_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist(),
        "accuracy": float(cv_scores.mean())
    }


# ---- Fusion Logic ----

def infer_from_perception(perception_output: PerceptionOutput) -> VictimState:
    """
    Fuse multi-modal perception data into a victim state assessment.
    
    CRITICAL: This function ONLY consumes perception_output.feature_vector
    and NEVER accesses raw sensor data or perception internals.
    
    Args:
        perception_output: PerceptionOutput from any perception module
    
    Returns:
        VictimState with responsiveness, vital signs, and confidence
    """
    global _model
    
    # Ensure model is loaded
    if _model is None:
        load_model()
    if _model is None:
        train_model(save=True)
    
    logger.debug(f"Processing perception output from {perception_output['track_id']}")
    
    victim_state: VictimState = {
        "presence_confirmed": False,
        "responsiveness": "UNCERTAIN",
        "vital_signs": "UNCERTAIN",
        "confidence_level": "LOW",
        "model_confidence": 0.0,
        "effective_confidence": 0.0,
        "fusion_explanation": [],
        "uncertain": False
    }
    
    # Check for UNCERTAIN state in perception
    if perception_output["uncertain"]:
        victim_state["uncertain"] = True
        victim_state["fusion_explanation"].append(
            f"Uncertain perception input: {', '.join(perception_output['uncertainty_flags'])}"
        )
        logger.info(f"Uncertain perception state: {perception_output['uncertainty_flags']}")
    
    # Human presence confirmation
    if not perception_output["detected"]:
        victim_state["fusion_explanation"].append("No human presence detected")
        logger.info("No human presence detected")
        return victim_state
    
    victim_state["presence_confirmed"] = True
    victim_state["fusion_explanation"].append(
        f"Human presence confirmed (detection_confidence={perception_output['detection_confidence']:.2f})"
    )
    
    # Use feature vector for ML inference
    # Ensure feature vector has 14 elements
    feature_vector = perception_output["feature_vector"]
    if len(feature_vector) != 14:
        logger.error(f"Expected 14 features, got {len(feature_vector)}")
        raise ValueError(f"Feature vector must have 14 elements, got {len(feature_vector)}")
    
    # Model prediction
    confidence_pred = _model.predict([feature_vector])[0]
    confidence_proba = _model.predict_proba([feature_vector])[0]
    
    confidence_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    victim_state["confidence_level"] = confidence_map[confidence_pred]
    victim_state["model_confidence"] = float(max(confidence_proba))
    
    logger.debug(f"Model prediction: {victim_state['confidence_level']} (proba={confidence_proba})")
    
    # Determine responsiveness from motion state
    motion_state = perception_output["motion_state"]
    if motion_state == "RESPONSIVE":
        victim_state["responsiveness"] = "RESPONSIVE"
        victim_state["fusion_explanation"].append(
            f"High responsiveness detected (motion_score={perception_output['motion_score']:.2f})"
        )
    elif motion_state == "LIMITED_RESPONSE":
        victim_state["responsiveness"] = "WEAK_RESPONSE"
        victim_state["fusion_explanation"].append(
            f"Limited responsiveness detected (motion_score={perception_output['motion_score']:.2f})"
        )
    elif motion_state == "NO_RESPONSE":
        victim_state["responsiveness"] = "UNRESPONSIVE"
        victim_state["fusion_explanation"].append(
            "No responsiveness detected"
        )
    else:  # UNCERTAIN
        victim_state["responsiveness"] = "UNCERTAIN"
        victim_state["fusion_explanation"].append(
            "Responsiveness uncertain due to poor motion detection"
        )
        victim_state["uncertain"] = True
    
    # Determine vital signs from thermal + breathing
    heat_present = perception_output["heat_present"]
    breathing = perception_output["breathing_proxy"]
    
    if heat_present and breathing == "DETECTED":
        victim_state["vital_signs"] = "STABLE_SIGNS"
        victim_state["fusion_explanation"].append(
            f"Stable vital indicators (heat={perception_output['thermal_intensity']:.2f}, breathing detected)"
        )
    elif heat_present and breathing in ["UNCERTAIN", "NOT_DETECTED"]:
        victim_state["vital_signs"] = "UNCERTAIN_SIGNS"
        victim_state["fusion_explanation"].append(
            f"Uncertain vital indicators (heat present but breathing uncertain)"
        )
        victim_state["uncertain"] = True
    else:
        victim_state["vital_signs"] = "WEAK_SIGNS"
        victim_state["fusion_explanation"].append(
            f"Weak vital indicators (heat_intensity={perception_output['thermal_intensity']:.2f})"
        )
    
    # Apply confidence reduction factors
    victim_state["effective_confidence"] = victim_state["model_confidence"] * perception_output["confidence_reduction_factor"]
    
    if perception_output["confidence_reduction_factor"] < 1.0:
        victim_state["fusion_explanation"].append(
            f"Confidence reduced due to: {', '.join(perception_output['uncertainty_flags'])}"
        )
        victim_state["uncertain"] = True
    
    victim_state["fusion_explanation"].append(
        f"ML model confidence: {victim_state['confidence_level']} ({victim_state['model_confidence']:.2f}), "
        f"effective: {victim_state['effective_confidence']:.2f}"
    )
    
    return victim_state


def fuse_signals(perception_outputs: List[PerceptionOutput]) -> VictimState:
    """
    Fuse multiple perception outputs into a single victim state.
    
    Args:
        perception_outputs: List of PerceptionOutput objects from perception layer
    
    Returns:
        VictimState with fused assessment
    """
    if not perception_outputs:
        logger.error("No perception outputs provided")
        raise ValueError("At least one perception output required")
    
    # For now, use the first (or most confident) perception output
    # In future, could implement weighted ensemble
    perception = perception_outputs[0] if isinstance(perception_outputs, list) else perception_outputs
    
    logger.debug(f"Fusing {len(perception_outputs) if isinstance(perception_outputs, list) else 1} perception output(s)")
    
    return infer_from_perception(perception)


def validate_perception_output(perception_output: dict) -> tuple[bool, str]:
    """
    Validate that input is a valid PerceptionOutput.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = {
        "detected", "detection_confidence",
        "motion_state", "motion_score", "motion_confidence",
        "heat_present", "thermal_intensity", "thermal_stability", "breathing_proxy",
        "thermal_confidence", "visibility_level", "signal_quality",
        "feature_vector", "uncertain", "uncertainty_flags", "confidence_reduction_factor"
    }
    
    missing = required_fields - set(perception_output.keys())
    if missing:
        error_msg = f"Missing required fields: {missing}"
        logger.warning(f"Validation failed: {error_msg}")
        return False, error_msg
    
    # Validate feature vector
    if not isinstance(perception_output["feature_vector"], (list, tuple)) or len(perception_output["feature_vector"]) != 14:
        error_msg = f"feature_vector must be list/tuple of 14 elements, got {len(perception_output.get('feature_vector', []))}"
        logger.warning(f"Validation failed: {error_msg}")
        return False, error_msg
    
    # Validate confidence values
    for conf_field in ["detection_confidence", "motion_confidence", "thermal_confidence", "signal_quality"]:
        value = perception_output.get(conf_field, 0)
        if not (0.0 <= value <= 1.0):
            error_msg = f"{conf_field} must be in [0, 1], got {value}"
            logger.warning(f"Validation failed: {error_msg}")
            return False, error_msg
    
    logger.debug("PerceptionOutput validation passed")
    return True, ""


# ---- Legacy Support ----

class SensorData(TypedDict):
    """Legacy type for backward compatibility."""
    human_detected: bool
    motion_level: Literal["NONE", "LOW", "HIGH"]
    heat_presence: Literal["NORMAL", "LOW"]
    breathing_detected: Literal["YES", "NO"]


def fuse_signals_legacy(sensor_data: SensorData) -> VictimState:
    """
    Legacy function for backward compatibility.
    Converts old SensorData to PerceptionOutput and calls infer_from_perception.
    
    DEPRECATED: Use infer_from_perception(perception_output) instead.
    """
    logger.warning("Using legacy fuse_signals - convert to PerceptionOutput in new code")
    
    # Convert legacy SensorData to PerceptionOutput
    motion_map = {"NONE": 0, "LOW": 1, "HIGH": 2}
    motion_state_map = {0: "NO_RESPONSE", 1: "LIMITED_RESPONSE", 2: "RESPONSIVE"}
    
    motion_numeric = motion_map.get(sensor_data["motion_level"], 1)
    motion_state = motion_state_map.get(motion_numeric, "UNCERTAIN")
    
    breathing_proxy = "DETECTED" if sensor_data["breathing_detected"] == "YES" else "NOT_DETECTED"
    heat_present = sensor_data["heat_presence"] == "NORMAL"
    
    # Build feature vector (14 elements)
    feature_vector = [
        float(1 if sensor_data["human_detected"] else 0),  # [0] detected
        0.7,                                                # [1] detection_confidence
        motion_numeric,                                     # [2] motion_state
        motion_numeric / 2.0,                               # [3] motion_score
        0.6,                                                # [4] motion_confidence
        float(1 if heat_present else 0),                    # [5] heat_present
        float(1 if heat_present else 0) * 0.7,              # [6] thermal_intensity
        0.5,                                                # [7] thermal_stability
        float(1 if breathing_proxy == "DETECTED" else 0),   # [8] breathing_proxy
        0.6,                                                # [9] breathing_confidence
        0.6,                                                # [10] thermal_confidence
        2,                                                  # [11] visibility_level (GOOD)
        0.65,                                               # [12] signal_quality
        0                                                   # [13] num_uncertainty_flags
    ]
    
    perception_output: PerceptionOutput = {
        "track_id": "legacy_0",
        "bbox": (0, 0, 0, 0),
        "frame_id": 0,
        "timestamp": 0.0,
        "detected": sensor_data["human_detected"],
        "detection_confidence": 0.7,
        "motion_state": motion_state,
        "motion_score": motion_numeric / 2.0,
        "motion_confidence": 0.6,
        "camera_motion_compensated": True,
        "heat_present": heat_present,
        "thermal_intensity": float(1 if heat_present else 0) * 0.7,
        "thermal_stability": 0.5,
        "breathing_proxy": breathing_proxy,
        "hypothermia_risk": "LOW",
        "thermal_confidence": 0.6,
        "visibility_level": "GOOD",
        "signal_quality": 0.65,
        "feature_vector": feature_vector,
        "uncertain": False,
        "uncertainty_flags": [],
        "confidence_reduction_factor": 1.0,
        "debug_info": None
    }
    
    return infer_from_perception(perception_output)
