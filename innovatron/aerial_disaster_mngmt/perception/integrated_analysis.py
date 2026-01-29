#!/usr/bin/env python3
"""
Integrated Perception System: Motion + Human Detection + Thermal Analysis
Combines all three detection modules on a single live webcam feed.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, '.')

from motion_analysis import MotionAnalyzer
from human_detection import HumanDetector
from thermal_analysis import ThermalAnalyzer

class IntegratedAnalyzer:
    """Combines motion, human detection, and thermal analysis."""
    
    def __init__(self):
        """Initialize all three analyzers."""
        print("Initializing Integrated Perception System...")
        print("-" * 60)
        
        self.motion_analyzer = MotionAnalyzer()
        print("[OK] Motion Analyzer initialized")
        
        self.human_detector = HumanDetector()
        print("[OK] Human Detector initialized")
        
        self.thermal_analyzer = ThermalAnalyzer()
        self.thermal_analyzer.set_thermal_mode("luminance")
        print("[OK] Thermal Analyzer initialized (mode: luminance)")
        
        self.frame_count = 0
        self.motion_alert_threshold = 0.7
        self.thermal_alert_threshold = 0.6
        self.grid_step = 20  # Grid spacing for motion visualization
        
        print("-" * 60)
        print("System ready for live analysis\n")
    
    def draw_motion_grid(self, frame, flow=None):
        """
        Draw clean motion visualization without grid lines.
        Shows only significant motion vectors for a cleaner appearance.
        
        Args:
            frame: input frame
            flow: optical flow (if None, no overlay)
        
        Returns:
            frame with minimal motion overlay
        """
        frame_copy = frame.copy()
        
        if flow is None:
            return frame_copy
        
        h, w = frame.shape[:2]
        step = self.grid_step
        
        # Generate grid point coordinates
        y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step].astype(int)
        
        # Extract flow values at grid points
        fx, fy = flow[y_coords, x_coords].T
        fx = fx.T
        fy = fy.T
        
        # Draw only significant motion arrows
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                x1, y1 = x_coords[i, j], y_coords[i, j]
                
                # Calculate motion magnitude
                magnitude = np.sqrt(fx[i, j]**2 + fy[i, j]**2)
                
                # Only draw if significant motion detected
                if magnitude > 1.0:
                    x2 = int(x1 + fx[i, j] * 2)
                    y2 = int(y1 + fy[i, j] * 2)
                    
                    # Color based on magnitude
                    intensity = min(int(magnitude * 15), 255)
                    color = (0, intensity, 255 - intensity)  # Green to Red
                    
                    # Draw arrow
                    cv2.arrowedLine(frame_copy, (x1, y1), (x2, y2), color, 2, tipLength=0.3)
        
        return frame_copy


    
    def run_live_analysis(self):
        """Run integrated analysis on live webcam feed."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot open webcam")
            return False
        
        print("Opening live feed... Press 'q' to quit, 's' to save screenshot\n")
        print("Frame | Motion Level | Motion Confidence | Humans | Thermal Risk | Events")
        print("-" * 80)
        
        prev_frame = None
        frame_bbox = None  # Will use full frame for motion analysis
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            self.frame_count += 1
            
            # Define full frame bbox [x1, y1, x2, y2] for motion/thermal analysis
            h, w = frame.shape[:2]
            frame_bbox = [0, 0, w, h]
            
            # ===== MOTION ANALYSIS =====
            motion_data = {}
            if prev_frame is not None:
                # Motion analyzer expects BGR frames
                motion_data = self.motion_analyzer.analyze_motion(prev_frame, frame, frame_bbox)
            prev_frame = frame.copy()
            
            motion_level = motion_data.get("motion_level", "UNKNOWN")
            motion_confidence = motion_data.get("confidence", 0.0)
            smoothed_score = motion_data.get("smoothed_score", 0.0)
            
            # ===== HUMAN DETECTION =====
            detections = self.human_detector.detect_humans(frame)
            num_humans = len(detections)
            
            # ===== THERMAL ANALYSIS =====
            thermal_data = self.thermal_analyzer.analyze_thermal(frame, frame_bbox)
            thermal_risk = "HIGH" if thermal_data.get("hypothermia_risk", False) else "LOW"
            thermal_confidence = thermal_data.get("confidence", 0.0)
            heat_present = thermal_data.get("heat_present", False)
            
            # ===== EVENT DETECTION =====
            events = []
            
            if motion_level == "HIGH" and motion_confidence > self.motion_alert_threshold:
                events.append("[MOTION] High motion detected")
            
            if num_humans > 0:
                events.append(f"[PERSONS] {num_humans} detected")
            
            if heat_present and thermal_data.get("heat_intensity_ratio", 0) > self.thermal_alert_threshold:
                events.append("[THERMAL] Heat anomaly detected")
            
            # ===== VISUALIZATION =====
            display_frame = frame.copy()
            
            # Draw motion grid with flow vectors
            if 'flow' in motion_data:
                display_frame = self.draw_motion_grid(display_frame, motion_data['flow'])
            else:
                display_frame = self.draw_motion_grid(display_frame)
            
            # Draw motion status
            cv2.putText(display_frame, f"Motion: {motion_level} ({motion_confidence:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw human detections
            for i, det in enumerate(detections):
                bbox = det.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                conf = det.get("confidence", 0.0)
                dist = det.get("distance_m", 0.0)
                
                # Color based on thermal data
                if heat_present:
                    color = (0, 0, 255)  # Red for thermal anomaly
                else:
                    color = (0, 255, 0)  # Green for normal
                
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(display_frame, f"Person {i+1} ({conf:.2f})",
                           (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(display_frame, f"Dist: {dist:.1f}m",
                           (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw thermal indicator
            thermal_color = (0, 0, 255) if thermal_risk == "HIGH" else (0, 165, 255)
            cv2.putText(display_frame, f"Thermal Risk: {thermal_risk} ({thermal_confidence:.2f})",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, thermal_color, 2)
            
            # Draw events
            if events:
                event_text = " | ".join(events)
                cv2.putText(display_frame, event_text,
                           (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 255), 2)
            
            # Draw frame info
            cv2.putText(display_frame, f"Frame: {self.frame_count}",
                       (display_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            # Print summary
            event_str = " | ".join(events) if events else "None"
            print(f"{self.frame_count:5d} | {motion_level:12s} | {motion_confidence:17.3f} | "
                  f"{num_humans:6d} | {thermal_risk:12s} | {event_str}")
            
            # Display
            cv2.imshow("Integrated Perception System", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('s'):
                filename = f"screenshot_{self.frame_count}.png"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def print_summary(self):
        """Print final statistics."""
        print("\n" + "=" * 80)
        print("Integrated Analysis Summary")
        print("=" * 80)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Motion analyzer history: {len(self.motion_analyzer.normalized_score_history)} frames")
        print(f"Human detector tracking: {len(self.human_detector.frame_history)} frame(s)")
        print(f"Thermal analyzer mode: {self.thermal_analyzer.thermal_mode}")
        print("=" * 80)


def main():
    """Main entry point."""
    analyzer = IntegratedAnalyzer()
    
    try:
        analyzer.run_live_analysis()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.print_summary()


if __name__ == "__main__":
    main()
