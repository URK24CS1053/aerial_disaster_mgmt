from typing import TypedDict, Tuple, Literal, Optional, List


# -------------------------
# Enumerations
# -------------------------

MotionState = Literal[
    "RESPONSIVE",
    "LIMITED_RESPONSE",
    "NO_RESPONSE",
    "UNCERTAIN"
]

BreathingProxy = Literal[
    "DETECTED",
    "NOT_DETECTED",
    "UNCERTAIN"
]

DetectorType = Literal[
    "YOLO",
    "HOG",
    "SIMULATED"
]

VisibilityLevel = Literal[
    "GOOD",
    "POOR",
    "UNKNOWN"
]

UncertaintyFlag = Literal[
    "LOW_DETECTION_CONFIDENCE",
    "GLOBAL_MOTION_DETECTED",
    "POOR_VISIBILITY",
    "INCONSISTENT_SIGNALS",
    "INSUFFICIENT_DATA",
    "THERMAL_ANOMALY"
]


# -------------------------
# Perception Output (ML-Ready Schema)
# -------------------------

class PerceptionOutput(TypedDict):
    """
    Unified perception output consumed exclusively by fusion_engine.py.
    
    This structure enforces:
    - Strict separation between perception and decision-making layers
    - ML-ready feature vectors for model inference
    - Explicit confidence and uncertainty tracking
    - Explainability without raw sensor data access
    - Graceful handling of UNCERTAIN states
    
    Feature Vector Composition (14 features, indices 0-13):
        [0]  - Human detected (binary: 0/1)
        [1]  - Detection confidence (0.0-1.0)
        [2]  - Motion state (0=NO_RESPONSE, 1=LIMITED_RESPONSE, 2=RESPONSIVE, 3=UNCERTAIN)
        [3]  - Motion score (0.0-1.0 normalized optical flow magnitude)
        [4]  - Motion confidence (0.0-1.0)
        [5]  - Heat present (binary: 0/1)
        [6]  - Thermal intensity (0.0-1.0 normalized)
        [7]  - Thermal stability (0.0-1.0 temporal consistency)
        [8]  - Breathing proxy (0=NOT_DETECTED, 1=DETECTED, 2=UNCERTAIN)
        [9]  - Breathing confidence (0.0-1.0)
        [10] - Thermal confidence (0.0-1.0)
        [11] - Visibility level (0=UNKNOWN, 1=POOR, 2=GOOD)
        [12] - Signal quality (0.0-1.0 aggregate reliability)
        [13] - Number of active uncertainty flags (0-6)
    """

    # ---- Identity & Tracking ----
    track_id: str                          # Unique identifier for this detection
    bbox: Tuple[int, int, int, int]        # (x, y, w, h) bounding box in pixels
    frame_id: int                          # Frame number
    timestamp: float                       # Unix timestamp

    # ---- Detection ----
    detected: bool                         # Whether human was detected
    detection_confidence: float            # 0–1, detector confidence

    # ---- Motion / Responsiveness ----
    motion_state: MotionState              # Qualitative motion state
    motion_score: float                    # 0–1, normalized optical flow magnitude
    motion_confidence: float               # 0–1, confidence in motion measurement
    camera_motion_compensated: bool        # Whether camera motion was filtered out

    # ---- Thermal / Pseudo-Thermal ----
    heat_present: bool                     # Whether heat signature detected
    thermal_intensity: float               # 0–1, normalized heat intensity
    thermal_stability: float               # 0–1, temporal consistency of thermal signal
    breathing_proxy: BreathingProxy        # Proxy for breathing via thermal/motion analysis
    hypothermia_risk: Literal["LOW", "HIGH"]  # Risk assessment
    thermal_confidence: float              # 0–1, confidence in thermal measurement

    # ---- Environmental Context ----
    visibility_level: VisibilityLevel      # Environmental visibility
    signal_quality: float                  # 0–1, aggregate signal reliability

    # ---- ML-Friendly Feature Vector ----
    feature_vector: List[float]            # Ordered 14-element numeric vector for ML inference
    
    # ---- Uncertainty Handling ----
    uncertain: bool                        # Whether state is uncertain (true if any flag is set)
    uncertainty_flags: List[UncertaintyFlag]  # Specific reasons for uncertainty
    confidence_reduction_factor: float     # 0–1, how much to reduce confidence (global motion, poor visibility, etc.)

    # ---- Debug / Logging (ignored by fusion_engine) ----
    debug_info: Optional[dict]             # Optional metadata for logging/analysis