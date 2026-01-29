# perception/perception_adapters.py
"""
Perception Layer Adapters

Adapters that wrap existing perception modules to output PerceptionOutput objects.
These adapters ensure backward compatibility while enforcing the new schema.

The adapters are thin wrappers that:
1. Call existing perception module methods
2. Extract results
3. Build PerceptionOutput with proper feature vectors and uncertainty handling
"""

from typing import List, Dict, Optional, Tuple
import sys
from datetime import datetime
import cv2
import numpy as np

# Import schema and utilities
sys.path.insert(0, '..')
from interfaces.perception_schema import PerceptionOutput
from perception_utils import create_perception_output, apply_confidence_reduction


class HumanDetectionAdapter:
    """Adapter for HumanDetector to output PerceptionOutput."""
    
    def __init__(self, detector):
        """Initialize with a HumanDetector instance."""
        self.detector = detector
        self.frame_id = 0
    
    def process_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> List[PerceptionOutput]:
        """
        Convert HumanDetector detections to PerceptionOutput objects.
        
        Args:
            frame: Input frame (for context)
            detections: List of detection dicts from detector
        
        Returns:
            List of PerceptionOutput objects
        """
        self.frame_id += 1
        timestamp = datetime.now().timestamp()
        outputs = []
        
        for i, det in enumerate(detections):
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # Extract detection info
            confidence = det.get("confidence", 0.0)
            detector_type = "YOLO" if self.detector.use_yolo else "HOG"
            
            # Create PerceptionOutput
            output = create_perception_output(
                track_id=det.get("track_id", f"human_{i}"),
                bbox=(x, y, w, h),
                frame_id=self.frame_id,
                timestamp=timestamp,
                detected=True,
                detection_confidence=confidence,
                motion_state="UNCERTAIN",  # Will be updated by MotionAnalysisAdapter
                motion_score=0.0,
                motion_confidence=0.0,
                camera_motion_compensated=False,
                heat_present=False,  # Will be updated by ThermalAnalysisAdapter
                thermal_intensity=0.0,
                thermal_stability=0.0,
                breathing_proxy="UNCERTAIN",
                hypothermia_risk="LOW",
                thermal_confidence=0.0,
                visibility_level="UNKNOWN",
                signal_quality=confidence,  # Use detection confidence as signal quality
                uncertainty_flags=[],
                confidence_reduction_factor=1.0,
                debug_info={
                    "detector_type": detector_type,
                    "distance_m": det.get("distance_m", 0.0),
                    "raw_detection": det
                }
            )
            
            outputs.append(output)
        
        return outputs


class ThermalAnalysisAdapter:
    """Adapter for ThermalAnalyzer to enhance PerceptionOutput objects."""
    
    def __init__(self, analyzer):
        """Initialize with a ThermalAnalyzer instance."""
        self.analyzer = analyzer
    
    def enhance_perception_output(
        self,
        perception_output: PerceptionOutput,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> PerceptionOutput:
        """
        Enhance PerceptionOutput with thermal analysis.
        
        Args:
            perception_output: Existing PerceptionOutput to enhance
            frame: Input frame
            bbox: [x1, y1, x2, y2] bounding box coordinates
        
        Returns:
            Enhanced PerceptionOutput with thermal data
        """
        try:
            # Run thermal analysis
            thermal_data = self.analyzer.analyze_thermal(frame, bbox)
            
            # Extract thermal info
            heat_present = thermal_data.get("heat_present", False)
            thermal_intensity = thermal_data.get("thermal_intensity", 0.0)
            thermal_stability = thermal_data.get("thermal_stability", 0.0)
            breathing = thermal_data.get("breathing_proxy", "UNCERTAIN")
            thermal_confidence = thermal_data.get("confidence", 0.0)
            hypothermia_risk = "HIGH" if thermal_data.get("hypothermia_risk", False) else "LOW"
            
            # Update perception output with thermal data
            perception_output["heat_present"] = heat_present
            perception_output["thermal_intensity"] = thermal_intensity
            perception_output["thermal_stability"] = thermal_stability
            perception_output["breathing_proxy"] = breathing
            perception_output["thermal_confidence"] = thermal_confidence
            perception_output["hypothermia_risk"] = hypothermia_risk
            
            # Update uncertainty flags if thermal data is unreliable
            if thermal_confidence < 0.3:
                if "THERMAL_ANOMALY" not in perception_output["uncertainty_flags"]:
                    perception_output["uncertainty_flags"].append("THERMAL_ANOMALY")
            
            # Rebuild feature vector with new thermal data
            from perception_utils import build_feature_vector
            perception_output["feature_vector"] = build_feature_vector(
                detected=perception_output["detected"],
                detection_confidence=perception_output["detection_confidence"],
                motion_state=perception_output["motion_state"],
                motion_score=perception_output["motion_score"],
                motion_confidence=perception_output["motion_confidence"],
                heat_present=heat_present,
                thermal_intensity=thermal_intensity,
                thermal_stability=thermal_stability,
                breathing_proxy=breathing,
                breathing_confidence=thermal_confidence,
                thermal_confidence=thermal_confidence,
                visibility_level=perception_output["visibility_level"],
                signal_quality=perception_output["signal_quality"],
                uncertainty_flags=perception_output["uncertainty_flags"]
            )
            
            return perception_output
        
        except Exception as e:
            # If thermal analysis fails, mark as uncertain but continue
            if "THERMAL_ANOMALY" not in perception_output["uncertainty_flags"]:
                perception_output["uncertainty_flags"].append("THERMAL_ANOMALY")
            return perception_output


class MotionAnalysisAdapter:
    """Adapter for MotionAnalyzer to enhance PerceptionOutput objects."""
    
    def __init__(self, analyzer):
        """Initialize with a MotionAnalyzer instance."""
        self.analyzer = analyzer
        self.prev_frame = None
    
    def enhance_perception_output(
        self,
        perception_output: PerceptionOutput,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> PerceptionOutput:
        """
        Enhance PerceptionOutput with motion analysis.
        
        Args:
            perception_output: Existing PerceptionOutput to enhance
            frame: Input frame
            bbox: [x1, y1, x2, y2] bounding box coordinates (or full frame)
        
        Returns:
            Enhanced PerceptionOutput with motion data
        """
        try:
            # Check for global camera motion
            if self.prev_frame is not None and self.prev_frame.shape == frame.shape:
                global_motion = self.analyzer.check_global_motion(self.prev_frame, frame)
                
                if global_motion.get("has_global_motion", False):
                    # Add uncertainty flag
                    if "GLOBAL_MOTION_DETECTED" not in perception_output["uncertainty_flags"]:
                        perception_output["uncertainty_flags"].append("GLOBAL_MOTION_DETECTED")
                    
                    # Apply confidence reduction
                    reduction_factor = global_motion.get("confidence_reduction", 1.0)
                    if reduction_factor < 1.0:
                        perception_output = apply_confidence_reduction(
                            perception_output,
                            "GLOBAL_MOTION_DETECTED",
                            reduction_factor
                        )
            
            # Analyze motion within bbox
            motion_data = self.analyzer.analyze_motion(self.prev_frame, frame, bbox) if self.prev_frame is not None else {}
            
            # Extract motion info
            motion_level = motion_data.get("motion_level", "UNCERTAIN")
            motion_score = motion_data.get("motion_score", 0.0)
            motion_confidence = motion_data.get("confidence", 0.0)
            camera_compensated = motion_data.get("camera_motion_compensated", False)
            
            # Convert motion level to motion state
            motion_state_map = {
                "NONE": "NO_RESPONSE",
                "LOW": "LIMITED_RESPONSE",
                "HIGH": "RESPONSIVE",
                "UNCERTAIN": "UNCERTAIN"
            }
            motion_state = motion_state_map.get(motion_level, "UNCERTAIN")
            
            # Update perception output
            perception_output["motion_state"] = motion_state
            perception_output["motion_score"] = motion_score
            perception_output["motion_confidence"] = motion_confidence
            perception_output["camera_motion_compensated"] = camera_compensated
            
            # Update signal quality
            perception_output["signal_quality"] = max(
                perception_output["signal_quality"],
                motion_confidence
            )
            
            # Store previous frame for next iteration
            self.prev_frame = frame.copy()
            
            # Rebuild feature vector with motion data
            from perception_utils import build_feature_vector
            perception_output["feature_vector"] = build_feature_vector(
                detected=perception_output["detected"],
                detection_confidence=perception_output["detection_confidence"],
                motion_state=motion_state,
                motion_score=motion_score,
                motion_confidence=motion_confidence,
                heat_present=perception_output["heat_present"],
                thermal_intensity=perception_output["thermal_intensity"],
                thermal_stability=perception_output["thermal_stability"],
                breathing_proxy=perception_output["breathing_proxy"],
                breathing_confidence=motion_confidence,
                thermal_confidence=perception_output["thermal_confidence"],
                visibility_level=perception_output["visibility_level"],
                signal_quality=perception_output["signal_quality"],
                uncertainty_flags=perception_output["uncertainty_flags"]
            )
            
            return perception_output
        
        except Exception as e:
            # If motion analysis fails, mark as uncertain but continue
            if "INCONSISTENT_SIGNALS" not in perception_output["uncertainty_flags"]:
                perception_output["uncertainty_flags"].append("INCONSISTENT_SIGNALS")
            return perception_output


def fuse_perception_adapters(
    frame: np.ndarray,
    human_adapter: HumanDetectionAdapter,
    thermal_adapter: ThermalAnalysisAdapter,
    motion_adapter: MotionAnalysisAdapter,
    detections: List[Dict]
) -> List[PerceptionOutput]:
    """
    Orchestrate all perception adapters to produce complete PerceptionOutput objects.
    
    Args:
        frame: Input frame
        human_adapter: HumanDetectionAdapter instance
        thermal_adapter: ThermalAnalysisAdapter instance
        motion_adapter: MotionAnalysisAdapter instance
        detections: List of detections from HumanDetector
    
    Returns:
        List of complete PerceptionOutput objects
    """
    # Step 1: Create PerceptionOutput from detections
    outputs = human_adapter.process_detections(frame, detections)
    
    # Step 2: Enhance with thermal analysis
    for i, output in enumerate(outputs):
        bbox = output["bbox"]
        x1, y1, w, h = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        outputs[i] = thermal_adapter.enhance_perception_output(
            output,
            frame,
            (x1, y1, int(x1 + w), int(y1 + h))
        )
    
    # Step 3: Enhance with motion analysis
    for i, output in enumerate(outputs):
        bbox = output["bbox"]
        x1, y1, w, h = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        outputs[i] = motion_adapter.enhance_perception_output(
            output,
            frame,
            (x1, y1, int(x1 + w), int(y1 + h))
        )
    
    return outputs
