# perception/human_detection.py

import cv2
import numpy as np

# Try to import YOLOv8, fall back to HOG if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: YOLOv8 not available ({type(e).__name__}), falling back to HOG detector")
    YOLO_AVAILABLE = False

class HumanDetector:
    def __init__(self, model_path=None, use_yolo=None):
        """
        Initialize detector for person detection only
        Supports both YOLOv8 and HOG fallback
        
        Args:
            model_path: Path to custom YOLO model (uses yolov8n.pt by default)
            use_yolo: Force use of YOLO (True) or HOG (False), auto-detect if None
        """
        # Determine which detector to use
        if use_yolo is None:
            use_yolo = YOLO_AVAILABLE
        
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        
        if self.use_yolo:
            # Load YOLOv8 model (yolov8n for nano - fastest, yolov8s/m/l for better accuracy)
            if model_path is None:
                model_path = "yolov8n.pt"
            
            self.model = YOLO(model_path)
            print("Using YOLOv8 detector")
        else:
            # Fallback to HOG detector
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("Using HOG detector (CPU fallback)")
        
        # COCO class IDs - Person is class 0
        self.person_class_id = 0
        
        # Camera parameters for distance estimation
        self.focal_length = 800
        self.known_height = 170  # Average human height in cm
        self.min_height = 30
        self.min_width = 20
        self.max_height_ratio = 3.5
        self.min_height_ratio = 1.2
        
        # Physical constraints for tracking
        self.max_size_change_ratio = 1.3  # Max size change between frames (30%)
        self.max_position_jump = 150  # Max pixels movement between frames
        self.frame_history = {}  # Track detections across frames
        self.history_retention_frames = 5  # Keep history for N frames
        
        # Bounding box smoothing parameters
        self.bbox_smoothing_enabled = True
        self.bbox_smoothing_window = 3  # Number of frames for smoothing window
        self.bbox_smoothing_alpha = 0.6  # Exponential moving average alpha (0.0-1.0)
        self.bbox_history = {}  # Track bbox coordinates across frames for smoothing

    def calculate_distance(self, bbox_height):
        """Calculate distance to person based on bbox height"""
        if bbox_height == 0:
            return float('inf')
        distance_cm = (self.focal_length * self.known_height) / bbox_height
        return distance_cm

    def is_valid_human_detection(self, x, y, w, h):
        """Validate detection based on aspect ratio and size constraints"""
        if w < self.min_width or h < self.min_height:
            return False
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < self.min_height_ratio or aspect_ratio > self.max_height_ratio:
            return False
        return True

    def _calculate_bbox_center(self, bbox):
        """Calculate center point of bounding box"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _calculate_bbox_area(self, bbox):
        """Calculate area of bounding box"""
        x, y, w, h = bbox
        return w * h

    def _euclidean_distance(self, p1, p2):
        """Calculate euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _has_plausible_size_change(self, prev_bbox, curr_bbox):
        """
        Check if size change between frames is plausible
        Rejects sudden size changes (e.g., false positives, occlusions)
        
        Args:
            prev_bbox: Previous detection bounding box (x, y, w, h)
            curr_bbox: Current detection bounding box (x, y, w, h)
        Returns:
            True if size change is within plausible limits
        """
        if prev_bbox is None:
            return True
        
        prev_area = self._calculate_bbox_area(prev_bbox)
        curr_area = self._calculate_bbox_area(curr_bbox)
        
        if prev_area == 0:
            return True
        
        size_ratio = curr_area / prev_area
        
        # Check if size change exceeds threshold
        if size_ratio > self.max_size_change_ratio or size_ratio < (1.0 / self.max_size_change_ratio):
            return False
        
        return True

    def _has_plausible_position(self, prev_bbox, curr_bbox):
        """
        Check if position change between frames is plausible
        Rejects sudden position jumps (impossible for slow-moving objects)
        
        Args:
            prev_bbox: Previous detection bounding box (x, y, w, h)
            curr_bbox: Current detection bounding box (x, y, w, h)
        Returns:
            True if position change is within plausible limits
        """
        if prev_bbox is None:
            return True
        
        prev_center = self._calculate_bbox_center(prev_bbox)
        curr_center = self._calculate_bbox_center(curr_bbox)
        
        # Calculate distance between centers
        distance = self._euclidean_distance(prev_center, curr_center)
        
        # Check if movement exceeds threshold
        if distance > self.max_position_jump:
            return False
        
        return True

    def _match_detection_with_history(self, detection):
        """
        Match current detection with closest detection from previous frame
        
        Args:
            detection: Current detection with bbox
        Returns:
            Previous detection bbox if found, None otherwise
        """
        curr_center = self._calculate_bbox_center(detection["bbox"])
        closest_match = None
        min_distance = float('inf')
        
        # Search through frame history for closest match
        for frame_id, frame_detections in self.frame_history.items():
            for prev_detection in frame_detections:
                prev_center = self._calculate_bbox_center(prev_detection["bbox"])
                distance = self._euclidean_distance(curr_center, prev_center)
                
                # Update closest match if this is nearer
                if distance < min_distance and distance < self.max_position_jump:
                    min_distance = distance
                    closest_match = prev_detection
        
        return closest_match

    def _apply_physical_constraints(self, detection):
        """
        Apply physical constraints to reject implausible detections
        
        Args:
            detection: Detection dict with bbox
        Returns:
            Tuple (is_valid, constraint_violations) where constraint_violations is a list of strings
        """
        violations = []
        
        # Try to match with previous detection
        matched_detection = self._match_detection_with_history(detection)
        
        # Check size plausibility
        if matched_detection is not None:
            if not self._has_plausible_size_change(matched_detection["bbox"], detection["bbox"]):
                violations.append("size_change_too_large")
        
        # Check position plausibility
        if matched_detection is not None:
            if not self._has_plausible_position(matched_detection["bbox"], detection["bbox"]):
                violations.append("position_jump_too_large")
        
        is_valid = len(violations) == 0
        
        return is_valid, violations, matched_detection

    def _cleanup_history(self, current_frame_id):
        """Remove old detections from history to save memory"""
        frames_to_remove = []
        for frame_id in self.frame_history.keys():
            if current_frame_id - frame_id > self.history_retention_frames:
                frames_to_remove.append(frame_id)
        
        for frame_id in frames_to_remove:
            del self.frame_history[frame_id]

    def _smooth_bbox_exponential(self, detection_id, current_bbox):
        """
        Apply exponential moving average smoothing to bounding box
        Reduces jitter while maintaining responsiveness to actual movement
        
        Args:
            detection_id: Unique identifier for tracking detection across frames
            current_bbox: Current bounding box (x, y, w, h)
        Returns:
            Smoothed bounding box (x, y, w, h)
        """
        if detection_id not in self.bbox_history:
            # First detection of this ID, initialize history
            self.bbox_history[detection_id] = {
                "history": [current_bbox],
                "last_smoothed": current_bbox
            }
            return current_bbox
        
        history_data = self.bbox_history[detection_id]
        last_smoothed = history_data["last_smoothed"]
        
        # Apply exponential moving average to each coordinate
        x, y, w, h = current_bbox
        prev_x, prev_y, prev_w, prev_h = last_smoothed
        
        # Formula: smoothed = alpha * current + (1 - alpha) * previous
        smoothed_x = int(self.bbox_smoothing_alpha * x + (1 - self.bbox_smoothing_alpha) * prev_x)
        smoothed_y = int(self.bbox_smoothing_alpha * y + (1 - self.bbox_smoothing_alpha) * prev_y)
        smoothed_w = int(self.bbox_smoothing_alpha * w + (1 - self.bbox_smoothing_alpha) * prev_w)
        smoothed_h = int(self.bbox_smoothing_alpha * h + (1 - self.bbox_smoothing_alpha) * prev_h)
        
        smoothed_bbox = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
        
        # Update history
        history_data["history"].append(current_bbox)
        history_data["last_smoothed"] = smoothed_bbox
        
        # Keep only recent history
        if len(history_data["history"]) > self.bbox_smoothing_window:
            history_data["history"].pop(0)
        
        return smoothed_bbox

    def _smooth_bbox_moving_average(self, detection_id, current_bbox):
        """
        Apply moving average smoothing to bounding box
        Uses average of last N frames for more stability
        
        Args:
            detection_id: Unique identifier for tracking detection across frames
            current_bbox: Current bounding box (x, y, w, h)
        Returns:
            Smoothed bounding box (x, y, w, h)
        """
        if detection_id not in self.bbox_history:
            # First detection of this ID, initialize history
            self.bbox_history[detection_id] = {
                "history": [current_bbox]
            }
            return current_bbox
        
        history_data = self.bbox_history[detection_id]
        history_data["history"].append(current_bbox)
        
        # Keep only last N frames
        if len(history_data["history"]) > self.bbox_smoothing_window:
            history_data["history"].pop(0)
        
        # Calculate average across history
        history_array = np.array(history_data["history"])
        smoothed_bbox = tuple(np.round(np.mean(history_array, axis=0)).astype(int))
        
        return smoothed_bbox

    def _apply_bbox_smoothing(self, detections, detection_ids=None):
        """
        Apply temporal smoothing to all detections in frame
        
        Args:
            detections: List of detection dicts with bbox
            detection_ids: Optional list of detection IDs (uses index if None)
        Returns:
            List of detections with smoothed bboxes
        """
        if not self.bbox_smoothing_enabled or len(detections) == 0:
            return detections
        
        smoothed_detections = []
        
        for idx, detection in enumerate(detections):
            # Use provided ID or auto-generate from index and confidence
            if detection_ids is not None and idx < len(detection_ids):
                det_id = detection_ids[idx]
            else:
                # Auto-generate ID from center position to maintain consistency
                det_id = f"det_{hash(str(detection['bbox'])) % 10000}"
            
            # Apply smoothing to bounding box
            original_bbox = detection["bbox"]
            smoothed_bbox = self._smooth_bbox_exponential(det_id, original_bbox)
            
            # Create new detection with smoothed bbox
            smoothed_detection = detection.copy()
            smoothed_detection["bbox"] = smoothed_bbox
            smoothed_detection["bbox_original"] = original_bbox  # Store original for reference
            smoothed_detection["detection_id"] = det_id
            
            # Recalculate distance based on smoothed bbox
            x, y, w, h = smoothed_bbox
            smoothed_detection["distance_cm"] = self.calculate_distance(h)
            smoothed_detection["distance_m"] = smoothed_detection["distance_cm"] / 100
            smoothed_detection["label"] = smoothed_detection["label"].replace(
                f"{detection['distance_m']:.2f}m",
                f"{smoothed_detection['distance_m']:.2f}m"
            )
            
            smoothed_detections.append(smoothed_detection)
        
        return smoothed_detections

    def _cleanup_bbox_history(self, active_detection_ids):
        """
        Remove smoothing history for detections no longer present
        
        Args:
            active_detection_ids: Set of detection IDs currently in frame
        """
        ids_to_remove = []
        for det_id in self.bbox_history.keys():
            if det_id not in active_detection_ids:
                ids_to_remove.append(det_id)
        
        for det_id in ids_to_remove:
            del self.bbox_history[det_id]

    def _group_detections(self, rects):
        """Group overlapping detections using NMS-like approach"""
        if len(rects) == 0:
            return rects
        rects = np.array(rects)
        x1 = rects[:, 0]
        y1 = rects[:, 1]
        x2 = x1 + rects[:, 2]
        y2 = y1 + rects[:, 3]
        areas = rects[:, 2] * rects[:, 3]
        order = np.argsort(areas)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w_inter = np.maximum(0, xx2 - xx1)
            h_inter = np.maximum(0, yy2 - yy1)
            inter = w_inter * h_inter
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union
            order = order[np.where(iou < 0.3)[0] + 1]
        grouped_rects = rects[np.array(keep)]
        return grouped_rects

    def _detect_humans_yolo(self, frame):
        """Detect humans using YOLOv8 - person class only"""
        detections = []
        
        # Run inference
        results = self.model(frame, conf=0.5, verbose=False)
        
        # Process results - filter for person class only (class_id = 0)
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Extract class ID
                class_id = int(box.cls[0])
                
                # Filter to person class only - discard all other classes
                if class_id != self.person_class_id:
                    continue
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1
                
                # Validate detection
                if not self.is_valid_human_detection(x, y, w, h):
                    continue
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Calculate distance
                distance_cm = self.calculate_distance(h)
                distance_m = distance_cm / 100
                
                detections.append({
                    "bbox": (x, y, w, h),
                    "confidence": confidence,
                    "distance_cm": distance_cm,
                    "distance_m": distance_m,
                    "class_id": class_id,
                    "class_name": "Person",
                    "label": f"Person (Conf: {confidence:.2f}, Dist: {distance_m:.2f}m)"
                })
        
        return detections

    def _detect_humans_hog(self, frame):
        """Detect humans using HOG detector (CPU fallback)"""
        detections = []
        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(16, 16), scale=1.05)
        rects_grouped = self._group_detections(rects)
        detection_id = 1
        
        for (x, y, w, h) in rects_grouped:
            if not self.is_valid_human_detection(x, y, w, h):
                continue
            distance_cm = self.calculate_distance(h)
            distance_m = distance_cm / 100
            detections.append({
                "bbox": (x, y, w, h),
                "confidence": 0.95,
                "distance_cm": distance_cm,
                "distance_m": distance_m,
                "class_id": 0,
                "class_name": "Person",
                "label": f"Person #{detection_id} ({distance_m:.2f}m)"
            })
            detection_id += 1
        return detections

    def detect_humans(self, frame, frame_id=None, apply_constraints=True, smooth_bbox=True):
        """
        Detect humans in frame using available detector (YOLOv8 or HOG fallback)
        Filters for person class only - discards all other classes
        Optionally applies physical constraints and temporal smoothing
        
        Args:
            frame: Input image/frame
            frame_id: Frame ID for tracking across time (auto-incremented if None)
            apply_constraints: If True, apply physical constraints filtering
            smooth_bbox: If True, apply temporal smoothing to bounding boxes
        Returns:
            List of detections with bbox, confidence, distance, and labels
        """
        if frame_id is None:
            frame_id = len(self.frame_history)
        
        # Get raw detections from appropriate detector
        if self.use_yolo:
            raw_detections = self._detect_humans_yolo(frame)
        else:
            raw_detections = self._detect_humans_hog(frame)
        
        # Apply physical constraints if enabled
        if apply_constraints and len(self.frame_history) > 0:
            validated_detections = []
            
            for detection in raw_detections:
                is_valid, violations, matched_detection = self._apply_physical_constraints(detection)
                
                if is_valid:
                    validated_detections.append(detection)
                else:
                    # Add violation info for debugging
                    detection["violations"] = violations
                    detection["rejected"] = True
            
            detections = validated_detections
        else:
            detections = raw_detections
        
        # Apply bounding box smoothing if enabled
        if smooth_bbox and len(detections) > 0:
            detections = self._apply_bbox_smoothing(detections)
            # Cleanup history for detections no longer present
            active_ids = {d.get("detection_id") for d in detections}
            self._cleanup_bbox_history(active_ids)
        
        # Store detections in history
        self.frame_history[frame_id] = detections
        
        # Cleanup old history
        self._cleanup_history(frame_id)
        
        return detections


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        exit(1)
    
    print("Initializing Human Detector...")
    detector = HumanDetector()
    print(f"Detector: {'YOLOv8' if detector.use_yolo else 'HOG (CPU fallback)'}")
    print("Person class filtering: ENABLED (all other classes discarded)")
    print("Press 'q' to quit")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        
        # Detect persons only
        detections = detector.detect_humans(frame)
        
        # Display info
        info_text = f"Frame: {frame_count} | Persons Detected: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(detections)} persons detected")
        
        # Draw detections
        for d in detections:
            x, y, w, h = d["bbox"]
            confidence = d["confidence"]
            distance_m = d.get("distance_m", 0)
            
            color = (0, 255, 0)  # Green for person
            label_text = f"Person ({confidence:.2f}) - {distance_m:.2f}m"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label with background
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-25), (x+text_size[0]+5, y), color, -1)
            cv2.putText(frame, label_text, (x+2, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        cv2.imshow("Human Detection - Person Class Only", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection complete")

