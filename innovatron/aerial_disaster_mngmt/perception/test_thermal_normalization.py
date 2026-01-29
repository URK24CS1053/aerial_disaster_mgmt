#!/usr/bin/env python3
"""
Test script for Thermal Analyzer with adaptive background subtraction
and lighting normalization features
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, '.')

from thermal_analysis import ThermalAnalyzer

def test_normalization_with_varying_lighting():
    """
    Simulate thermal analysis with varying lighting conditions
    Shows how the normalized metrics adapt to lighting changes
    """
    print("\n" + "=" * 70)
    print("THERMAL ANALYZER - ADAPTIVE NORMALIZATION TEST")
    print("=" * 70)
    
    # Initialize analyzer
    ta = ThermalAnalyzer(intensity_threshold=0.15)
    print("\n✓ ThermalAnalyzer initialized with adaptive normalization")
    print(f"  - Background subtraction: {ta.enable_background_subtraction}")
    print(f"  - Global normalization: {ta.global_normalization_enabled}")
    print(f"  - Background alpha (EMA): {ta.background_alpha}")
    
    # Test 1: Simulate bright lighting conditions
    print("\n" + "-" * 70)
    print("Test 1: BRIGHT LIGHTING CONDITIONS")
    print("-" * 70)
    
    # Create synthetic bright frame
    bright_frame = np.ones((240, 320, 3), dtype=np.uint8) * 220
    # Create bright background
    bbox_bright = (50, 50, 80, 100)
    
    # Simulate a warmer (higher intensity) region
    bright_frame[50:150, 50:130] = 230
    
    result_bright = ta.analyze_thermal(bright_frame, bbox_bright, detection_id="person_1")
    
    print(f"  Frame statistics:")
    print(f"    - Raw bbox intensity: {result_bright['bbox_intensity']:.1f}")
    print(f"    - Raw background intensity: {result_bright['background_intensity']:.1f}")
    print(f"    - Heat intensity ratio: {result_bright['heat_intensity_ratio']:.4f}")
    print(f"    - Normalized difference: {result_bright['normalized_difference']:.4f}")
    print(f"    - BG z-score: {result_bright['bg_zscore']:.4f}")
    print(f"    - Adaptive threshold: {result_bright['adaptive_threshold']:.4f}")
    print(f"    - Heat present: {result_bright['heat_present']}")
    print(f"    - Confidence: {result_bright['confidence']:.4f}")
    
    # Test 2: Simulate dark lighting conditions
    print("\n" + "-" * 70)
    print("Test 2: DARK LIGHTING CONDITIONS")
    print("-" * 70)
    
    # Create synthetic dark frame
    dark_frame = np.ones((240, 320, 3), dtype=np.uint8) * 50
    bbox_dark = (50, 50, 80, 100)
    
    # Simulate a warmer (higher intensity) region relative to dark background
    dark_frame[50:150, 50:130] = 80
    
    result_dark = ta.analyze_thermal(dark_frame, bbox_dark, detection_id="person_2")
    
    print(f"  Frame statistics:")
    print(f"    - Raw bbox intensity: {result_dark['bbox_intensity']:.1f}")
    print(f"    - Raw background intensity: {result_dark['background_intensity']:.1f}")
    print(f"    - Heat intensity ratio: {result_dark['heat_intensity_ratio']:.4f}")
    print(f"    - Normalized difference: {result_dark['normalized_difference']:.4f}")
    print(f"    - BG z-score: {result_dark['bg_zscore']:.4f}")
    print(f"    - Adaptive threshold: {result_dark['adaptive_threshold']:.4f}")
    print(f"    - Heat present: {result_dark['heat_present']}")
    print(f"    - Confidence: {result_dark['confidence']:.4f}")
    
    # Test 3: Simulate temporal tracking (background model adaptation)
    print("\n" + "-" * 70)
    print("Test 3: TEMPORAL BACKGROUND MODEL ADAPTATION")
    print("-" * 70)
    print("Simulating 5 consecutive frames with same detection")
    
    ta2 = ThermalAnalyzer(intensity_threshold=0.15)
    frame_sequence = np.ones((240, 320, 3), dtype=np.uint8) * 100
    bbox = (50, 50, 80, 100)
    frame_sequence[50:150, 50:130] = 150  # Warmer region
    
    print(f"\n{'Frame':<8} {'BG Intensity':<15} {'BG Model':<15} {'Norm Diff':<12} {'Heat':<8}")
    print("-" * 70)
    
    for frame_num in range(1, 6):
        result = ta2.analyze_thermal(frame_sequence, bbox, detection_id="person_tracked")
        bg_model = ta2.background_history.get("person_tracked", {}).get("intensity", 0)
        
        print(f"{frame_num:<8} {result['background_intensity']:<15.2f} "
              f"{bg_model:<15.2f} {result['normalized_difference']:<12.4f} "
              f"{str(result['heat_present']):<8}")
    
    # Test 4: Frame statistics computation
    print("\n" + "-" * 70)
    print("Test 4: FRAME STATISTICS FOR ADAPTIVE THRESHOLDING")
    print("-" * 70)
    
    # Create frame with varied lighting
    varied_frame = np.random.randint(80, 200, (240, 320, 3), dtype=np.uint8)
    stats = ta._compute_frame_statistics(varied_frame)
    
    print(f"  Global frame statistics:")
    print(f"    - Mean: {stats['mean']:.2f}")
    print(f"    - Std Dev: {stats['std']:.2f}")
    print(f"    - Min: {stats['min']:.2f}")
    print(f"    - Max: {stats['max']:.2f}")
    print(f"    - 75th percentile: {stats['percentile_75']:.2f}")
    print(f"    - 90th percentile: {stats['percentile_90']:.2f}")
    
    # Test 5: Multiple objects with independent background models
    print("\n" + "-" * 70)
    print("Test 5: MULTIPLE DETECTIONS WITH INDEPENDENT MODELS")
    print("-" * 70)
    
    ta3 = ThermalAnalyzer(intensity_threshold=0.15)
    test_frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
    test_frame[30:130, 30:110] = 180  # Person 1 - warmer
    test_frame[140:240, 140:220] = 90   # Person 2 - cooler
    
    person1_bbox = (30, 30, 80, 100)
    person2_bbox = (140, 140, 80, 100)
    
    result_p1 = ta3.analyze_thermal(test_frame, person1_bbox, detection_id="person_1")
    result_p2 = ta3.analyze_thermal(test_frame, person2_bbox, detection_id="person_2")
    
    print(f"\nPerson 1 (warmer region):")
    print(f"  - Intensity: {result_p1['bbox_intensity']:.1f}")
    print(f"  - Normalized diff: {result_p1['normalized_difference']:.4f}")
    print(f"  - Heat present: {result_p1['heat_present']}")
    print(f"  - Confidence: {result_p1['confidence']:.4f}")
    
    print(f"\nPerson 2 (cooler region):")
    print(f"  - Intensity: {result_p2['bbox_intensity']:.1f}")
    print(f"  - Normalized diff: {result_p2['normalized_difference']:.4f}")
    print(f"  - Heat present: {result_p2['heat_present']}")
    print(f"  - Confidence: {result_p2['confidence']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ All normalization tests completed successfully!")
    print("=" * 70 + "\n")


def test_live_with_normalization():
    """Test normalization on live webcam feed"""
    print("\nTesting normalization on live webcam feed...")
    print("Press 'q' to quit, 's' to save metrics\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam - skipping live test")
        return
    
    ta = ThermalAnalyzer()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Simulate detection of center region
        h, w = frame.shape[:2]
        bbox = (w//4, h//4, w//2, h//2)
        
        result = ta.analyze_thermal(frame, bbox, detection_id="test_person")
        
        # Display on frame
        cv2.putText(frame, f"Heat: {result['heat_present']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Norm: {result['normalized_difference']:.4f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {result['confidence']:.4f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bbox
        x, y, w_bbox, h_bbox = bbox
        cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), (0, 255, 0), 2)
        
        cv2.imshow("Thermal Normalization Live Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    # Run normalization tests
    test_normalization_with_varying_lighting()
    
    # Optional: run live test
    # Uncomment the line below to test on live webcam
    # test_live_with_normalization()
