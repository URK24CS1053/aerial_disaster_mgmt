#!/usr/bin/env python3
# Test script to verify human detection is working

import cv2
import sys
sys.path.insert(0, '.')
from human_detection import HumanDetector

def test_detection():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return False
    
    print("✓ Webcam opened")
    
    detector = HumanDetector()
    print("✓ Detector initialized")
    print(f"  Filter settings:")
    print(f"    Min size: {detector.min_width}x{detector.min_height}")
    print(f"    Height ratio: {detector.min_height_ratio}-{detector.max_height_ratio}")
    print()
    
    frame_count = 0
    total_detections = 0
    
    print("Running detection for 50 frames...")
    print("-" * 50)
    
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections = detector.detect_humans(frame)
        total_detections += len(detections)
        
        print(f"Frame {frame_count:2d}: {len(detections)} human(s) detected", end="")
        if len(detections) > 0:
            for d in detections:
                dist = d["distance_m"]
                print(f" | Distance: {dist:.2f}m", end="")
        print()
    
    cap.release()
    
    print("-" * 50)
    print(f"Summary:")
    print(f"  Frames: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg per frame: {total_detections/frame_count:.2f}")
    return True

if __name__ == "__main__":
    success = test_detection()
    exit(0 if success else 1)
