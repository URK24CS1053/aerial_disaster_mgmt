# Thermal Analysis Module - Complete Reference

## Overview

The `ThermalAnalyzer` class provides comprehensive thermal intensity analysis with adaptive background subtraction and multi-method normalization for robust heat detection across varying lighting and camera conditions.

## Class Initialization

```python
from thermal_analysis import ThermalAnalyzer

# Default initialization
ta = ThermalAnalyzer(
    intensity_threshold=0.15,              # Heat detection threshold
    breathing_variance_threshold=0.05      # Breathing motion threshold
)
```

## Core Attributes

### Intensity Detection
- `intensity_threshold` (float): Relative intensity difference for heat detection (default: 0.15)
- `breathing_variance_threshold` (float): Variance threshold for breathing (default: 0.05)

### RGB to Thermal Conversion
- `thermal_mode` (str): "luminance" | "red_channel" | "hsv_value"
- `use_adaptive_histogram` (bool): Apply CLAHE enhancement (default: True)
- `gaussian_blur_kernel` (int): Smoothing kernel size (default: 5)
- `clahe_clip_limit` (float): CLAHE contrast enhancement (default: 2.0)
- `clahe_tile_size` (int): CLAHE tile grid size (default: 8)

### Adaptive Background Subtraction (NEW)
- `enable_background_subtraction` (bool): Enable EMA background model (default: True)
- `background_alpha` (float): EMA adaptation rate 0-1 (default: 0.7)
  - Higher (0.8-0.9): Stable, slow adaptation
  - Lower (0.5-0.6): Responsive, fast adaptation
- `background_history` (dict): Per-detection background models

### Global Frame Normalization (NEW)
- `global_normalization_enabled` (bool): Adapt threshold to frame stats (default: True)
- `global_frame_stats` (dict): Cached frame statistics
- `intensity_percentile` (int): Percentile for robust statistics (default: 75)

### History Tracking
- `intensity_history` (dict): Per-detection breathing history
- `max_history_frames` (int): History buffer size (default: 10)

## Primary Methods

### analyze_thermal()

Main analysis method - analyzes thermal intensity in ROI.

```python
result = ta.analyze_thermal(
    thermal_frame,           # BGR/RGB frame or grayscale thermal
    bbox,                    # (x, y, width, height)
    detection_id=None        # Optional tracking ID
)
```

**Returns**: Dictionary with 19 fields

```python
{
    # Core Detection Fields
    'heat_present': bool,                    # Heat detected
    'heat_intensity_ratio': float,           # Raw ratio (may be negative)
    'hypothermia_risk': str,                 # 'LOW' | 'HIGH'
    'confidence': float,                     # 0-1 confidence score
    
    # Intensity Measurements
    'bbox_intensity': float,                 # ROI mean intensity (0-255)
    'background_intensity': float,           # Background mean intensity
    'bbox_intensity_min': float,             # ROI minimum
    'bbox_intensity_max': float,             # ROI maximum
    'bbox_intensity_std': float,             # ROI standard deviation
    
    # NEW - Normalization Metrics
    'normalized_difference': float,          # Background-subtracted ratio
    'bg_zscore': float,                      # Background z-score (-3 to +3)
    'percentile_normalized': float,          # Percentile-based ranking
    'adaptive_threshold': float,             # Current adaptive threshold
    
    # Thermal Characteristics
    'thermal_gradient': float,               # Std / Max (0-1)
    'hotspot_ratio': float,                  # Percent above mean (0-1)
    
    # Motion Detection
    'breathing_detected': bool,              # Temporal variance detected
    'breathing_variance': float,             # Variance magnitude
    
    # Metadata
    'detection_id': str or None,             # Tracking ID
    'thermal_source': str                    # "rgb_to_luminance" | "thermal" | etc.
}
```

### set_thermal_mode()

Configure RGB to thermal conversion method.

```python
ta.set_thermal_mode("luminance")    # Standard grayscale (default)
ta.set_thermal_mode("red_channel")  # Extract red channel
ta.set_thermal_mode("hsv_value")    # Extract HSV value channel
```

### cleanup_history()

Remove tracking history when detection disappears.

```python
ta.cleanup_history(detection_id)
```

## Normalization Methods (NEW)

### _compute_frame_statistics()

Calculate global frame statistics for adaptive normalization.

```python
stats = ta._compute_frame_statistics(thermal_frame)
# Returns:
{
    'mean': float,              # Average intensity
    'std': float,               # Standard deviation
    'min': float,               # Minimum
    'max': float,               # Maximum
    'percentile_75': float,     # 75th percentile
    'percentile_90': float      # 90th percentile
}
```

### _update_background_model()

Update exponential moving average background model.

```python
updated_bg = ta._update_background_model(
    detection_id,
    current_background_intensity
)
# Returns: Updated background intensity
```

### _normalize_intensity()

Apply multiple normalization methods to intensity values.

```python
norms = ta._normalize_intensity(
    bbox_intensity,
    background_intensity,
    global_stats=stats
)
# Returns:
{
    'raw_difference': float,           # Simple subtraction
    'relative_difference': float,      # Ratio-based
    'bbox_zscore': float,              # Z-score (bbox)
    'bg_zscore': float,                # Z-score (background)
    'percentile_normalized': float     # Percentile ranking
}
```

## Usage Examples

### Example 1: Basic Heat Detection

```python
import cv2
from thermal_analysis import ThermalAnalyzer

ta = ThermalAnalyzer()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    bbox = (100, 100, 80, 100)
    
    result = ta.analyze_thermal(frame, bbox, detection_id="person_1")
    
    if result['heat_present']:
        print(f"Heat detected! Confidence: {result['confidence']:.2f}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### Example 2: Adaptive Detection with Tuning

```python
# Create with custom thresholds
ta = ThermalAnalyzer(intensity_threshold=0.20)

# Fast adaptation to changing lighting
ta.background_alpha = 0.5

# Enable all normalization features
ta.enable_background_subtraction = True
ta.global_normalization_enabled = True

# Analyze with tracking
result = ta.analyze_thermal(frame, bbox, detection_id="tracked_person")

# Check multiple signals
if result['heat_present']:
    print(f"Normalized: {result['normalized_difference']:.4f}")
    print(f"Z-score: {result['bg_zscore']:.4f}")
    print(f"Percentile: {result['percentile_normalized']:.4f}")
    print(f"Threshold: {result['adaptive_threshold']:.4f}")
```

### Example 3: Multiple Detections

```python
ta = ThermalAnalyzer()
detections = [
    {'id': 'person_1', 'bbox': (50, 50, 80, 100)},
    {'id': 'person_2', 'bbox': (200, 150, 70, 120)}
]

for det in detections:
    result = ta.analyze_thermal(
        frame, 
        det['bbox'], 
        detection_id=det['id']
    )
    print(f"{det['id']}: Heat={result['heat_present']}, "
          f"Conf={result['confidence']:.2f}")
```

### Example 4: Cleanup on Detection Loss

```python
tracked_people = {}

while True:
    ret, frame = cap.read()
    current_detections = detect_people(frame)
    
    # Add new detections
    for det in current_detections:
        det_id = det['detection_id']
        tracked_people[det_id] = det
    
    # Analyze thermal
    for det_id, det_info in tracked_people.items():
        result = ta.analyze_thermal(frame, det_info['bbox'], det_id)
        print(f"{det_id}: Heat={result['heat_present']}")
    
    # Cleanup removed detections
    removed = set(tracked_people.keys()) - set(d['detection_id'] for d in current_detections)
    for det_id in removed:
        ta.cleanup_history(det_id)
        del tracked_people[det_id]
```

## Normalization Explanation

### Why Multiple Normalization Methods?

1. **Background Subtraction**: Local adaptation to nearby lighting
2. **Z-Score**: Global frame statistics for consistent scaling
3. **Percentile-Based**: Robust to outliers and extreme values
4. **Adaptive Threshold**: Adjusts sensitivity based on noise level

### Which to Use?

- **Small outdoor variations**: Use background subtraction only
- **Indoor with stable lighting**: Use z-score normalization
- **Mixed environments**: Use all methods (default)
- **High-noise scenes**: Rely on adaptive threshold

## Tuning Guide

### Scenario 1: Security Camera (Sensitive)
```python
ta.intensity_threshold = 0.10       # Lower threshold
ta.background_alpha = 0.6           # Fast adaptation
ta.enable_background_subtraction = True
ta.global_normalization_enabled = True
```
**Result**: Detects subtle heat changes, may have false positives

### Scenario 2: Surveillance (Stable)
```python
ta.intensity_threshold = 0.20       # Higher threshold
ta.background_alpha = 0.85          # Slow adaptation
ta.enable_background_subtraction = True
ta.global_normalization_enabled = True
```
**Result**: Fewer false positives, misses subtle changes

### Scenario 3: Outdoor (Dynamic Lighting)
```python
ta.intensity_threshold = 0.15       # Balanced
ta.background_alpha = 0.7           # Moderate
ta.global_normalization_enabled = True
ta.use_adaptive_histogram = True
```
**Result**: Handles sun/shadow transitions

### Scenario 4: Indoor Office (Fixed)
```python
ta.intensity_threshold = 0.15
ta.background_alpha = 0.9           # Stable model
ta.enable_background_subtraction = True
ta.global_normalization_enabled = False
```
**Result**: Stable in consistent lighting

## Performance Characteristics

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Frame statistics (640x480) | 1-2 | Percentile computation |
| Background update | <0.1 | Per detection |
| Normalization | 0.5 | Per detection |
| RGB to thermal | 2-3 | With CLAHE |
| Full analysis (2 persons) | 5-7 | Total end-to-end |

## Integration with Detection Modules

### With human_detection.py
```python
from human_detection import HumanDetector
from thermal_analysis import ThermalAnalyzer

detector = HumanDetector()
thermal = ThermalAnalyzer()

frame = cv2.imread('image.jpg')
humans = detector.detect_humans(frame)

for human in humans:
    thermal_result = thermal.analyze_thermal(
        frame,
        human['bbox'],
        detection_id=human['detection_id']
    )
    human['thermal_risk'] = thermal_result['hypothermia_risk']
    human['thermal_confidence'] = thermal_result['confidence']
```

### With motion_analysis.py
```python
from motion_analysis import MotionAnalyzer
from thermal_analysis import ThermalAnalyzer

motion = MotionAnalyzer()
thermal = ThermalAnalyzer()

# If motion detected in region
if motion_result['motion_level'] == 'HIGH':
    # Verify with thermal
    thermal_result = thermal.analyze_thermal(frame, roi_bbox)
    if thermal_result['heat_present']:
        print("Confirmed: Motion + thermal anomaly")
```

## Troubleshooting

### Issue: False Positives in Bright Light
**Solution**: Increase intensity_threshold or use_adaptive_histogram

### Issue: Missed Detections in Dark Scenes
**Solution**: Decrease intensity_threshold, use red_channel mode

### Issue: Unstable Detections
**Solution**: Increase background_alpha for more stable model

### Issue: Too Slow to Adapt
**Solution**: Decrease background_alpha for faster response

## Testing

Run comprehensive tests:
```bash
python test_thermal_normalization.py
```

Test with live video:
```bash
python integrated_analysis.py
```

## References

- Exponential Moving Average: https://en.wikipedia.org/wiki/Moving_average
- Z-Score: https://en.wikipedia.org/wiki/Standard_score
- CLAHE: OpenCV documentation
- Background Subtraction: Computer Vision fundamentals

## Files

- `thermal_analysis.py`: Main implementation (600+ lines)
- `test_thermal_normalization.py`: Comprehensive tests
- `THERMAL_NORMALIZATION_GUIDE.py`: Detailed guide with examples
- `integrated_analysis.py`: Multi-module integration example

## Version

Thermal Analysis Module v2.0 with Adaptive Normalization
- Date: January 2026
- Status: Production Ready
- Backward Compatible: Yes
