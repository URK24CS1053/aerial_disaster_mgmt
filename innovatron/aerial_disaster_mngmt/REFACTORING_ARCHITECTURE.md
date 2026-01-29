# Refactored SAR System - Integration Architecture

## Overview

The SAR (Search & Rescue) Victim Detection & Urgency Scoring system has been refactored to enforce **strict separation between perception and decision-making layers**, exclusively through the **PerceptionOutput schema**.

### Key Refactoring Goals Achieved

✅ **Exclusive PerceptionOutput Schema**
- Perception layer outputs ONLY PerceptionOutput objects
- Fusion engine consumes ONLY PerceptionOutput
- Decision-making layer has NO access to raw sensor data
- Feature vectors are ML-ready and normalized (14-element format)

✅ **Strict Layer Separation**
- No implicit dependencies between modules
- Data flows unidirectionally: Perception → Fusion → Urgency → Logging
- Each layer has a single, well-defined contract (input schema, output schema)

✅ **ML-Based Inference**
- Fusion engine performs inference on `feature_vector` exclusively
- No raw sensor data or perception internals accessed
- Probabilistic inference with confidence levels

✅ **Graceful Uncertainty Handling**
- Explicit `UNCERTAIN` states with specific flags
- `confidence_reduction_factor` applies across pipeline
- All uncertainty propagates to urgency scoring

✅ **Full Explainability**
- `fusion_explanation` chains decisions with reasoning
- `uncertainty_flags` document why decisions are uncertain
- All decisions logged with full audit trail

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ MAIN.PY - SOLE INTEGRATION POINT                               │
│ (Orchestrates perception → fusion → urgency → logging)         │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ PERCEPTION LAYER (Independent)                                  │
│                                                                  │
│ ├─ human_detection.py    → Detections + Confidence             │
│ ├─ thermal_analysis.py   → Heat Signatures + Breathing          │
│ ├─ motion_analysis.py    → Motion States + Scores               │
│ ├─ perception_adapters.py → Converts to PerceptionOutput        │
│ └─ perception_utils.py   → Feature vector building              │
│                                                                  │
│ ✓ Returns ONLY PerceptionOutput                                 │
│ ✓ No raw data leakage to downstream layers                      │
│ ✓ Feature vectors (14 normalized floats)                        │
│ ✓ Explicit uncertainty & confidence reduction factors           │
└─────────────────────────────────────────────────────────────────┘
                               ↓
                      [PerceptionOutput]
        (14-element feature_vector + uncertainty flags)
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ FUSION ENGINE - DECISION LAYER (fusion_engine.py)              │
│                                                                  │
│ ├─ infer_from_perception()                                      │
│ │  ├─ Validates PerceptionOutput schema                         │
│ │  ├─ Extracts feature_vector ONLY (no raw data)               │
│ │  ├─ ML inference: RandomForestClassifier                      │
│ │  ├─ Handles UNCERTAIN states gracefully                       │
│ │  └─ Returns VictimState with explainability                   │
│                                                                  │
│ ├─ fuse_signals() [for multi-modal fusion]                      │
│ ├─ validate_perception_output() [schema enforcement]            │
│ ├─ train_model() [for offline training]                         │
│                                                                  │
│ ✓ NO raw sensor data access                                     │
│ ✓ NO perception internals access                                │
│ ✓ Feature vector is ONLY input to ML model                      │
│ ✓ Confidence reduction applied across decision                  │
└─────────────────────────────────────────────────────────────────┘
                               ↓
                       [VictimState]
        (presence, responsiveness, vital_signs, confidence)
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ URGENCY SCORING LAYER (urgency_scoring.py)                     │
│                                                                  │
│ ├─ assign_urgency()                                              │
│ │  ├─ Consumes ONLY VictimState                                 │
│ │  ├─ Environmental risk factors                                │
│ │  └─ Returns UrgencyResult with reasoning                      │
│                                                                  │
│ ✓ No perception data access                                     │
│ ✓ Full explainability via reason field                          │
└─────────────────────────────────────────────────────────────────┘
                               ↓
                     [UrgencyResult]
         (urgency_level, reasons)
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ LOGGING & AUDIT (logger_config.py)                              │
│ Records full decision chain with timestamps & explanations      │
└─────────────────────────────────────────────────────────────────┘
```

---

## PerceptionOutput Schema

Located: `interfaces/perception_schema.py`

### Structure

```python
class PerceptionOutput(TypedDict):
    # Identity & Tracking
    track_id: str
    bbox: Tuple[int, int, int, int]        # (x, y, w, h)
    frame_id: int
    timestamp: float
    
    # Detection
    detected: bool
    detection_confidence: float             # 0.0-1.0
    
    # Motion / Responsiveness
    motion_state: MotionState               # "RESPONSIVE", "LIMITED_RESPONSE", "NO_RESPONSE", "UNCERTAIN"
    motion_score: float                     # 0.0-1.0 (normalized optical flow)
    motion_confidence: float                # 0.0-1.0
    camera_motion_compensated: bool
    
    # Thermal / Pseudo-Thermal
    heat_present: bool
    thermal_intensity: float                # 0.0-1.0
    thermal_stability: float                # 0.0-1.0 (temporal consistency)
    breathing_proxy: BreathingProxy         # "DETECTED", "NOT_DETECTED", "UNCERTAIN"
    hypothermia_risk: Literal["LOW", "HIGH"]
    thermal_confidence: float               # 0.0-1.0
    
    # Environmental Context
    visibility_level: VisibilityLevel       # "GOOD", "POOR", "UNKNOWN"
    signal_quality: float                   # 0.0-1.0 (aggregate reliability)
    
    # ML-Friendly Feature Vector (14 elements)
    feature_vector: List[float]
    # [0]  = human detected (0 or 1)
    # [1]  = detection confidence (0.0-1.0)
    # [2]  = motion state (0-3)
    # [3]  = motion score (0.0-1.0)
    # [4]  = motion confidence (0.0-1.0)
    # [5]  = heat present (0 or 1)
    # [6]  = thermal intensity (0.0-1.0)
    # [7]  = thermal stability (0.0-1.0)
    # [8]  = breathing proxy (0-2)
    # [9]  = breathing confidence (0.0-1.0)
    # [10] = thermal confidence (0.0-1.0)
    # [11] = visibility level (0-2)
    # [12] = signal quality (0.0-1.0)
    # [13] = number of uncertainty flags (0-6)
    
    # Uncertainty Handling
    uncertain: bool
    uncertainty_flags: List[UncertaintyFlag]
    # Possible flags:
    # - "LOW_DETECTION_CONFIDENCE"
    # - "GLOBAL_MOTION_DETECTED"
    # - "POOR_VISIBILITY"
    # - "INCONSISTENT_SIGNALS"
    # - "INSUFFICIENT_DATA"
    # - "THERMAL_ANOMALY"
    confidence_reduction_factor: float      # 0.0-1.0 (cumulative reduction)
    
    # Debug / Logging (ignored by ML)
    debug_info: Optional[dict]
```

### Feature Vector Details

The 14-element feature vector is designed to be:
- **ML-Ready**: Normalized and numeric
- **Self-Contained**: No need for raw data access by downstream
- **Complete**: Captures all modalities (detection, motion, thermal)
- **Uncertainty-Aware**: Includes count of uncertainty flags
- **Consistent**: Always 14 elements, same ordering

---

## Key Files Changed

### 1. `interfaces/perception_schema.py`
- **Refactored PerceptionOutput** with detailed feature vector spec
- Added `UncertaintyFlag` enum
- Added `confidence_reduction_factor` field
- Documented 14-element feature vector composition

### 2. `decision_making/fusion_engine.py`
- **Completely refactored** to consume ONLY PerceptionOutput
- New function: `infer_from_perception(perception_output: PerceptionOutput) → VictimState`
- Removes all raw sensor data processing
- Feature vector extraction at input
- ML-based inference with confidence tracking
- Graceful UNCERTAIN state handling
- Legacy support: `fuse_signals_legacy()` for old SensorData format

### 3. `decision_making/main.py`
- **New role**: Sole orchestration point
- `orchestrate_perception_to_urgency()` - modern pipeline
- `orchestrate_legacy_pipeline()` - backward compatibility
- Enhanced logging and audit trail
- Architecture visualization
- Full refactoring documentation

### 4. `perception/perception_adapters.py` (NEW)
- `HumanDetectionAdapter` - wraps HumanDetector
- `ThermalAnalysisAdapter` - enhances with thermal data
- `MotionAnalysisAdapter` - enhances with motion data
- `fuse_perception_adapters()` - orchestrates all adapters
- Converts legacy detection formats to PerceptionOutput

### 5. `perception/perception_utils.py` (NEW)
- `create_perception_output()` - factory function
- `build_feature_vector()` - constructs 14-element vector
- `apply_confidence_reduction()` - handles uncertainty
- `validate_perception_output()` - schema enforcement
- Enum conversion helpers (motion_state, breathing_proxy, etc.)

---

## Integration Patterns

### Pattern 1: Modern Pipeline (Recommended)

```python
from decision_making.main import orchestrate_perception_to_urgency
from perception.perception_adapters import (
    HumanDetectionAdapter, 
    ThermalAnalysisAdapter, 
    MotionAnalysisAdapter,
    fuse_perception_adapters
)

# Create perception outputs (using adapters)
perception_outputs = fuse_perception_adapters(
    frame=video_frame,
    human_adapter=human_adapter,
    thermal_adapter=thermal_adapter,
    motion_adapter=motion_adapter,
    detections=detections
)

# Orchestrate decision pipeline
result = orchestrate_perception_to_urgency(
    perception_outputs=perception_outputs,
    environment_risk="LOW"
)

# Access decisions
for decision in result["decisions"]:
    urgency_level = decision["urgency"]["urgency_level"]
    confidence = decision["effective_confidence"]
    explanation = decision["victim_state"]["fusion_explanation"]
```

### Pattern 2: Direct Fusion Engine Use

```python
from decision_making.fusion_engine import infer_from_perception
from interfaces.perception_schema import PerceptionOutput

# perception_output is a PerceptionOutput dict
victim_state = infer_from_perception(perception_output)

# victim_state includes:
# - presence_confirmed
# - responsiveness
# - vital_signs
# - confidence_level (HIGH, MEDIUM, LOW)
# - effective_confidence (0.0-1.0 after reduction)
# - fusion_explanation (list of reasoning steps)
# - uncertain (bool)
```

### Pattern 3: Legacy Support (Backward Compatibility)

```python
from decision_making.main import orchestrate_legacy_pipeline

# Old SensorData format
sensor_data = {
    "human_detected": True,
    "motion_level": "HIGH",
    "heat_presence": "NORMAL",
    "breathing_detected": "YES"
}

# Orchestrate with legacy pipeline
result = orchestrate_legacy_pipeline(
    sensor_data=sensor_data,
    environment_risk="LOW"
)
```

---

## Uncertainty Handling

### UNCERTAIN States Propagation

```
Perception Layer (uncertainty_flags + confidence_reduction_factor)
         ↓
Fusion Engine (checks uncertain flag, applies reduction)
         ↓
VictimState (uncertain=True, reduced effective_confidence)
         ↓
Urgency Scoring (escalates based on effective_confidence)
         ↓
Logging (records all uncertainty sources)
```

### Confidence Reduction Example

```
Original model_confidence: 0.85
Global motion detected: confidence_reduction_factor = 0.8
Effective confidence = 0.85 * 0.8 = 0.68

Fusion explanation: "Global motion detected - confidence reduced"
Uncertainty flags: ["GLOBAL_MOTION_DETECTED"]
Uncertain: True
```

---

## Testing & Validation

### Running the System

```bash
cd decision_making
python main.py
```

This will:
1. Display system architecture
2. Show system information
3. Run model evaluation
4. Execute victim scenarios (legacy path for backward compatibility)
5. Generate audit logs

### Running Tests

```bash
# Test fusion engine with PerceptionOutput
python fusion_engine.py

# Test urgency scoring
python urgency_scoring.py
```

### Validating PerceptionOutput

```python
from perception.perception_utils import validate_perception_output

is_valid, error_msg = validate_perception_output(perception_dict)
if not is_valid:
    print(f"Validation failed: {error_msg}")
```

---

## Backward Compatibility

The system maintains full backward compatibility:

1. **Legacy SensorData Format** still works via `fuse_signals_legacy()`
2. **Old main.py functionality** preserved via `orchestrate_legacy_pipeline()`
3. **Existing tests** still pass (using legacy path)
4. **Gradual migration** - can mix old and new code

However, **new code should use PerceptionOutput exclusively**.

---

## Benefits of Refactoring

### 1. **Strict Separation of Concerns**
- Perception layer: Extract signals
- Fusion layer: Make decisions
- Urgency layer: Prioritize
- Each layer is independent and testable

### 2. **No Implicit Dependencies**
- Fusion engine cannot access raw sensors
- Urgency layer cannot access perception details
- Clear contracts between layers (via TypedDict schemas)

### 3. **ML-Ready Architecture**
- Feature vectors are standardized
- Confidence values are normalized
- Uncertainty is explicit
- Easy to swap ML models

### 4. **Explainability**
- `fusion_explanation` documents all reasoning
- `uncertainty_flags` explain why decisions are uncertain
- Full audit trail via logging
- Transparent decision-making

### 5. **Scalability**
- Easy to add new perception modules
- Easy to add new ML models
- Easy to implement multi-modal fusion
- Feature vector format is extensible

### 6. **Testability**
- Each layer can be tested independently
- Mock PerceptionOutput for testing fusion
- Mock VictimState for testing urgency
- Deterministic feature vectors

---

## Future Enhancements

### Planned
1. **Multi-Modal Fusion** - Ensemble multiple perception outputs
2. **Temporal Filtering** - Track state across frames
3. **Active Learning** - Request clarification on uncertain states
4. **Feature Engineering** - Learn optimal feature combinations
5. **A/B Testing** - Compare ML models on same perception data

### Possible
1. **Real-time Optimization** - Adaptive thresholds
2. **Federated Learning** - Decentralized model training
3. **Explainable AI** - SHAP values for feature importance
4. **Transfer Learning** - Domain adaptation

---

## File Structure

```
aerial_disaster_mngmt/
├── interfaces/
│   └── perception_schema.py          # PerceptionOutput schema
├── perception/
│   ├── human_detection.py            # (unchanged core)
│   ├── thermal_analysis.py           # (unchanged core)
│   ├── motion_analysis.py            # (unchanged core)
│   ├── integrated_analysis.py        # (unchanged core)
│   ├── perception_adapters.py        # NEW - Layer adapters
│   └── perception_utils.py           # NEW - Utilities
└── decision_making/
    ├── main.py                       # REFACTORED - Orchestrator
    ├── fusion_engine.py              # REFACTORED - Decision layer
    ├── urgency_scoring.py            # (unchanged, uses VictimState)
    ├── logger_config.py              # (unchanged)
    └── sensor_training_data.csv      # (unchanged)
```

---

## Summary

The refactored SAR system achieves **strict layer separation through PerceptionOutput** while maintaining:
- Full backward compatibility
- Explainability and auditability
- ML-ready feature vectors
- Graceful uncertainty handling
- No implicit dependencies

The architecture is now **scalable, testable, and maintainable**.

---

## Questions?

For architecture questions, refer to:
- `interfaces/perception_schema.py` - Schema documentation
- `decision_making/main.py` - Orchestration patterns
- `perception/perception_adapters.py` - Adapter examples
- `decision_making/fusion_engine.py` - ML inference details
