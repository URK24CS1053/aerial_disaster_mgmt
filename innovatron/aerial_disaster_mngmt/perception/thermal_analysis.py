# perception/thermal_analysis.py

import cv2
import numpy as np

class ThermalAnalyzer:
    def __init__(self, intensity_threshold=0.15, breathing_variance_threshold=0.05):
        """
        Initialize thermal analyzer with intensity-based detection
        
        Args:
            intensity_threshold: Relative intensity difference threshold (0.0-1.0)
                Difference = (bbox_intensity - bg_intensity) / bg_intensity
            breathing_variance_threshold: Variance threshold for breathing detection
        """
        self.intensity_threshold = intensity_threshold
        self.breathing_variance_threshold = breathing_variance_threshold
        self.intensity_history = {}  # Track intensity across frames
        self.max_history_frames = 10  # Keep last N frames for breathing detection
        
        # RGB to thermal conversion parameters
        self.thermal_mode = "luminance"  # "luminance", "red_channel", "hsv_value"
        self.use_adaptive_histogram = True  # Enhance contrast for better heat detection
        self.gaussian_blur_kernel = 5  # For smoothing pseudo-thermal data
        self.clahe_clip_limit = 2.0  # CLAHE contrast enhancement
        self.clahe_tile_size = 8  # CLAHE tile size
        
        # Background subtraction for adaptive lighting normalization
        self.background_history = {}  # Track background intensity across frames per detection
        self.global_frame_stats = {}  # Track global frame statistics for normalization
        self.background_alpha = 0.7  # Exponential moving average weight for background
        self.enable_background_subtraction = True  # Enable adaptive background normalization
        self.global_normalization_enabled = True  # Normalize by frame-wide statistics
        self.intensity_percentile = 75  # Use 75th percentile for robust background estimation

    def _rgb_to_luminance(self, rgb_frame):
        """
        Convert RGB frame to luminance (grayscale)
        Uses standard BT.709 color space conversion
        
        Args:
            rgb_frame: BGR or RGB frame (3 channels)
        Returns:
            Grayscale image (single channel)
        """
        if len(rgb_frame.shape) == 2:
            return rgb_frame  # Already grayscale
        
        # Convert BGR to grayscale using standard weights
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        return gray

    def _rgb_to_red_channel(self, rgb_frame):
        """
        Extract red channel from RGB frame
        Red channel often correlates with thermal heat in visible light
        
        Args:
            rgb_frame: BGR frame
        Returns:
            Red channel as grayscale
        """
        if len(rgb_frame.shape) == 2:
            return rgb_frame  # Already grayscale
        
        # Extract red channel (index 2 in BGR format)
        red_channel = rgb_frame[:, :, 2]
        return red_channel

    def _rgb_to_hsv_value(self, rgb_frame):
        """
        Extract Value channel from HSV color space
        V channel represents brightness and can indicate heat
        
        Args:
            rgb_frame: BGR frame
        Returns:
            Value channel as grayscale
        """
        if len(rgb_frame.shape) == 2:
            return rgb_frame  # Already grayscale
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        # Extract V channel (index 2 in HSV)
        v_channel = hsv[:, :, 2]
        return v_channel

    def _rgb_to_pseudo_thermal(self, rgb_frame, thermal_mode=None):
        """
        Convert RGB frame to pseudo-thermal representation
        Simulates thermal camera output using visible light data
        
        Args:
            rgb_frame: BGR or RGB frame from webcam
            thermal_mode: Conversion method ("luminance", "red_channel", "hsv_value")
                         Uses self.thermal_mode if None
        Returns:
            Pseudo-thermal grayscale image
        """
        if thermal_mode is None:
            thermal_mode = self.thermal_mode
        
        # Select conversion method
        if thermal_mode == "red_channel":
            thermal = self._rgb_to_red_channel(rgb_frame)
        elif thermal_mode == "hsv_value":
            thermal = self._rgb_to_hsv_value(rgb_frame)
        else:  # luminance (default)
            thermal = self._rgb_to_luminance(rgb_frame)
        
        # Apply adaptive histogram equalization for better contrast
        if self.use_adaptive_histogram:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
            )
            thermal = clahe.apply(thermal)
        
        # Apply Gaussian blur to smooth the thermal representation
        # This reduces noise and makes patterns more visible
        if self.gaussian_blur_kernel > 0 and self.gaussian_blur_kernel % 2 == 1:
            thermal = cv2.GaussianBlur(thermal, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 1.0)
        
        return thermal

    def set_thermal_mode(self, mode):
        """
        Set RGB to thermal conversion mode
        
        Args:
            mode: "luminance", "red_channel", or "hsv_value"
        """
        valid_modes = ["luminance", "red_channel", "hsv_value"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid thermal mode. Use one of: {valid_modes}")
        self.thermal_mode = mode
    
    def _compute_frame_statistics(self, thermal_frame, frame_id=None):
        """
        Compute global frame statistics for adaptive normalization
        Handles changing lighting and exposure conditions
        
        Args:
            thermal_frame: Input thermal frame
            frame_id: Optional frame identifier for history
        Returns:
            dict with frame statistics (mean, std, min, max, percentile)
        """
        if thermal_frame is None or thermal_frame.size == 0:
            return {
                "mean": 128,
                "std": 32,
                "min": 0,
                "max": 255,
                "percentile_75": 192,
                "percentile_90": 230
            }
        
        # Convert to grayscale if needed
        if len(thermal_frame.shape) == 3:
            gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal_frame
        
        # Compute statistics
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        percentile_75 = float(np.percentile(gray, 75))
        percentile_90 = float(np.percentile(gray, 90))
        
        stats = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "percentile_75": percentile_75,
            "percentile_90": percentile_90
        }
        
        # Store in history if frame_id provided
        if frame_id is not None:
            self.global_frame_stats[frame_id] = stats
        
        return stats
    
    def _update_background_model(self, detection_id, background_intensity):
        """
        Update exponential moving average background model
        Adapts to gradual lighting changes
        
        Args:
            detection_id: Unique detection identifier
            background_intensity: Current background intensity measurement
        Returns:
            Updated background intensity (adapted value)
        """
        if detection_id not in self.background_history:
            self.background_history[detection_id] = {
                "intensity": background_intensity,
                "update_count": 1
            }
            return background_intensity
        
        # Exponential moving average: new_val = alpha * current + (1-alpha) * previous
        prev_bg = self.background_history[detection_id]["intensity"]
        updated_bg = (self.background_alpha * background_intensity + 
                     (1 - self.background_alpha) * prev_bg)
        
        self.background_history[detection_id]["intensity"] = updated_bg
        self.background_history[detection_id]["update_count"] += 1
        
        return updated_bg
    
    def _normalize_intensity(self, bbox_intensity, background_intensity, 
                            global_stats=None, frame_id=None):
        """
        Normalize intensity values using background subtraction
        Adapts to changing lighting and camera exposure
        
        Args:
            bbox_intensity: Intensity in bounding box
            background_intensity: Local background intensity
            global_stats: Optional global frame statistics for additional normalization
            frame_id: Optional frame identifier for history
        Returns:
            dict with normalized values and metrics
        """
        # Basic background subtraction
        raw_difference = bbox_intensity - background_intensity
        
        # Normalized difference (relative to background)
        if background_intensity > 0:
            relative_difference = raw_difference / background_intensity
        else:
            relative_difference = 0.0
        
        # Global normalization (if frame statistics available)
        if self.global_normalization_enabled and global_stats is not None:
            frame_mean = global_stats.get("mean", 128)
            frame_std = global_stats.get("std", 32)
            
            # Standardize using z-score: (value - mean) / std
            if frame_std > 0:
                bbox_zscore = (bbox_intensity - frame_mean) / frame_std
                bg_zscore = (background_intensity - frame_mean) / frame_std
            else:
                bbox_zscore = 0.0
                bg_zscore = 0.0
        else:
            bbox_zscore = 0.0
            bg_zscore = 0.0
        
        # Adaptive threshold based on global statistics
        if global_stats is not None:
            # Use percentile-based normalization for robustness
            frame_percentile_75 = global_stats.get("percentile_75", 192)
            frame_percentile_90 = global_stats.get("percentile_90", 230)
            
            # Normalize to percentile range
            if frame_percentile_90 > frame_percentile_75:
                percentile_normalized = ((bbox_intensity - frame_percentile_75) / 
                                        (frame_percentile_90 - frame_percentile_75))
            else:
                percentile_normalized = 0.0
        else:
            percentile_normalized = 0.0
        
        return {
            "raw_difference": float(raw_difference),
            "relative_difference": float(relative_difference),
            "bbox_zscore": float(bbox_zscore),
            "bg_zscore": float(bg_zscore),
            "percentile_normalized": float(percentile_normalized)
        }

    def _get_bbox_region(self, frame, bbox, padding=0):
        """
        Extract region of interest from frame
        
        Args:
            frame: Input frame (BGR or grayscale)
            bbox: Bounding box (x, y, w, h)
            padding: Extra padding around bbox in pixels
        Returns:
            Region array or None if invalid
        """
        x, y, w, h = bbox
        
        # Apply padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        
        region = frame[y:y_end, x:x_end]
        
        if region.size == 0:
            return None
        
        return region

    def _get_background_region(self, frame, bbox, bg_margin=20):
        """
        Extract background region surrounding the bbox
        Used to establish baseline intensity
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            bg_margin: Margin around bbox for background sampling
        Returns:
            Background region array or None if invalid
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        
        # Create background mask around bbox - use top and bottom regions
        bg_regions = []
        
        # Top region (above bbox)
        if y >= bg_margin:
            bg_y_start = max(0, y - bg_margin)
            bg_y_end = y
            bg_x_start = max(0, x)
            bg_x_end = min(w_frame, x + w)
            top_region = frame[bg_y_start:bg_y_end, bg_x_start:bg_x_end]
            if top_region.size > 0:
                bg_regions.append(top_region)
        
        # Bottom region (below bbox)
        if y + h + bg_margin <= h_frame:
            bg_y_start = y + h
            bg_y_end = min(h_frame, y + h + bg_margin)
            bg_x_start = max(0, x)
            bg_x_end = min(w_frame, x + w)
            bottom_region = frame[bg_y_start:bg_y_end, bg_x_start:bg_x_end]
            if bottom_region.size > 0:
                bg_regions.append(bottom_region)
        
        # Combine all background regions
        if len(bg_regions) == 0:
            return None
        
        # Vertically stack regions with matching width
        background = np.vstack(bg_regions) if len(bg_regions) > 1 else bg_regions[0]
        return background

    def _calculate_intensity(self, region):
        """
        Calculate average intensity of region
        
        Args:
            region: Region array (can be BGR or grayscale)
        Returns:
            Average intensity (0-255)
        """
        if region is None or region.size == 0:
            return 0
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Calculate mean intensity
        intensity = np.mean(gray)
        
        return intensity

    def _detect_breathing(self, detection_id, current_intensity):
        """
        Detect breathing motion based on intensity variance over time
        Breathing causes rhythmic intensity changes as chest rises/falls
        
        Args:
            detection_id: Unique detection identifier
            current_intensity: Current intensity value
        Returns:
            Tuple (breathing_detected, variance) where variance is temporal variance
        """
        if detection_id not in self.intensity_history:
            self.intensity_history[detection_id] = []
        
        # Add current intensity to history
        self.intensity_history[detection_id].append(current_intensity)
        
        # Keep only recent history
        if len(self.intensity_history[detection_id]) > self.max_history_frames:
            self.intensity_history[detection_id].pop(0)
        
        # Need at least 3 samples to detect breathing
        if len(self.intensity_history[detection_id]) < 3:
            return False, 0.0
        
        # Calculate variance in intensity over time
        intensity_array = np.array(self.intensity_history[detection_id])
        variance = np.var(intensity_array)
        
        # Breathing detected if variance exceeds threshold
        breathing_detected = variance > self.breathing_variance_threshold
        
        return breathing_detected, float(variance)

    def analyze_thermal(self, thermal_frame, bbox, detection_id=None):
        """
        Analyze thermal/intensity characteristics from thermal ROI
        Accepts thermal frame (actual thermal or RGB webcam) and computes features
        Uses adaptive background subtraction for lighting invariance
        
        Args:
            thermal_frame: Thermal frame (grayscale or BGR)
                          OR RGB/BGR webcam frame (auto-converts to pseudo-thermal)
            bbox: Bounding box (x, y, w, h) defining ROI
            detection_id: Optional unique ID for tracking breathing across frames
        Returns:
            dict with comprehensive thermal analysis results including normalized metrics
        """
        if thermal_frame is None or thermal_frame.size == 0:
            return {
                "heat_present": False,
                "heat_intensity_ratio": 0.0,
                "bbox_intensity": 0,
                "background_intensity": 0,
                "bbox_intensity_min": 0,
                "bbox_intensity_max": 0,
                "bbox_intensity_std": 0.0,
                "breathing_detected": False,
                "breathing_variance": 0.0,
                "hypothermia_risk": "UNKNOWN",
                "confidence": 0.0,
                "thermal_gradient": 0.0,
                "hotspot_ratio": 0.0,
                "detection_id": detection_id,
                "thermal_source": "unknown",
                "normalized_difference": 0.0,
                "bg_zscore": 0.0,
                "percentile_normalized": 0.0,
                "adaptive_threshold": 0.15
            }
        
        # Check if input is RGB (3 channels) and convert to pseudo-thermal
        if len(thermal_frame.shape) == 3 and thermal_frame.shape[2] == 3:
            # RGB/BGR frame detected, convert to pseudo-thermal
            thermal_frame = self._rgb_to_pseudo_thermal(thermal_frame)
            thermal_source = f"rgb_to_{self.thermal_mode}"
        else:
            thermal_source = "thermal" if len(thermal_frame.shape) == 2 else "unknown"
        
        # Compute global frame statistics for adaptive normalization
        frame_stats = self._compute_frame_statistics(thermal_frame)
        
        # Extract bbox and background regions from thermal frame
        bbox_region = self._get_bbox_region(thermal_frame, bbox, padding=2)
        background_region = self._get_background_region(thermal_frame, bbox, bg_margin=20)
        
        if bbox_region is None:
            return {
                "heat_present": False,
                "heat_intensity_ratio": 0.0,
                "bbox_intensity": 0,
                "background_intensity": 0,
                "bbox_intensity_min": 0,
                "bbox_intensity_max": 0,
                "bbox_intensity_std": 0.0,
                "breathing_detected": False,
                "breathing_variance": 0.0,
                "hypothermia_risk": "UNKNOWN",
                "confidence": 0.0,
                "thermal_gradient": 0.0,
                "hotspot_ratio": 0.0,
                "detection_id": detection_id,
                "thermal_source": thermal_source,
                "normalized_difference": 0.0,
                "bg_zscore": 0.0,
                "percentile_normalized": 0.0,
                "adaptive_threshold": 0.15
            }
        
        # Convert to grayscale for intensity calculation
        if len(bbox_region.shape) == 3:
            bbox_gray = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY)
        else:
            bbox_gray = bbox_region
        
        # Calculate bbox thermal features
        bbox_intensity = self._calculate_intensity(bbox_region)
        bbox_intensity_min = float(np.min(bbox_gray))
        bbox_intensity_max = float(np.max(bbox_gray))
        bbox_intensity_std = float(np.std(bbox_gray))
        
        # Calculate background intensity
        background_intensity = self._calculate_intensity(background_region) if background_region is not None else bbox_intensity
        
        # Apply adaptive background subtraction
        if self.enable_background_subtraction and detection_id is not None:
            background_intensity = self._update_background_model(detection_id, background_intensity)
        
        # Normalize intensity values using background subtraction
        normalization_metrics = self._normalize_intensity(
            bbox_intensity, 
            background_intensity,
            global_stats=frame_stats
        )
        
        # Use normalized metrics for detection
        relative_difference = normalization_metrics["relative_difference"]
        normalized_difference = normalization_metrics["relative_difference"]
        bg_zscore = normalization_metrics["bg_zscore"]
        percentile_normalized = normalization_metrics["percentile_normalized"]
        
        # Adaptive threshold: adjust based on frame statistics
        adaptive_threshold = self.intensity_threshold
        if frame_stats["std"] > 0:
            # Increase threshold in noisy frames, decrease in clean frames
            noise_factor = min(1.5, frame_stats["std"] / 64.0)  # Normalize to typical std
            adaptive_threshold = self.intensity_threshold * noise_factor
        
        # Calculate thermal gradient (how much variation in bbox)
        # Normalized by max intensity to account for scaling
        if bbox_intensity_max > 0:
            thermal_gradient = bbox_intensity_std / bbox_intensity_max
        else:
            thermal_gradient = 0.0
        
        # Calculate hotspot ratio (how much of bbox is above mean)
        threshold = bbox_intensity
        hotspot_pixels = np.sum(bbox_gray >= threshold)
        total_pixels = bbox_gray.size
        hotspot_ratio = hotspot_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Determine if heat is present using normalized metric
        heat_present = normalized_difference > adaptive_threshold
        
        # Detect breathing if heat is present
        breathing_detected = False
        breathing_variance = 0.0
        
        if heat_present and detection_id is not None:
            breathing_detected, breathing_variance = self._detect_breathing(detection_id, bbox_intensity)
        
        # Determine hypothermia risk based on normalized metrics
        if heat_present:
            hypothermia_risk = "LOW"
            # Use zscore for confidence (higher zscore = higher confidence)
            confidence = min(1.0, abs(bg_zscore) / 3.0)  # 3 sigma confidence
        else:
            hypothermia_risk = "HIGH"
            confidence = min(1.0, abs(normalized_difference) / adaptive_threshold)
        
        return {
            "heat_present": heat_present,
            "heat_intensity_ratio": float(relative_difference),
            "bbox_intensity": float(bbox_intensity),
            "background_intensity": float(background_intensity),
            "bbox_intensity_min": bbox_intensity_min,
            "bbox_intensity_max": bbox_intensity_max,
            "bbox_intensity_std": bbox_intensity_std,
            "breathing_detected": breathing_detected,
            "breathing_variance": breathing_variance,
            "hypothermia_risk": hypothermia_risk,
            "confidence": float(confidence),
            "thermal_gradient": float(thermal_gradient),
            "hotspot_ratio": float(hotspot_ratio),
            "detection_id": detection_id,
            "thermal_source": thermal_source,
            "normalized_difference": float(normalized_difference),
            "bg_zscore": float(bg_zscore),
            "percentile_normalized": float(percentile_normalized),
            "adaptive_threshold": float(adaptive_threshold)
        }

    def cleanup_history(self, detection_id):
        """
        Remove intensity and background history for a detection no longer present
        
        Args:
            detection_id: Detection ID to remove from history
        """
        if detection_id in self.intensity_history:
            del self.intensity_history[detection_id]
        if detection_id in self.background_history:
            del self.background_history[detection_id]


# Quick test
if __name__ == "__main__":
    print("Thermal Analysis Module")
    print("=" * 50)
    print("This module analyzes thermal/intensity characteristics")
    print("based on pixel intensity within bounding boxes.")
    print()
    print("Features:")
    print("  - Heat detection via intensity comparison")
    print("  - Background normalization")
    print("  - Breathing detection via temporal variance")
    print("  - Hypothermia risk assessment")
    print("=" * 50)