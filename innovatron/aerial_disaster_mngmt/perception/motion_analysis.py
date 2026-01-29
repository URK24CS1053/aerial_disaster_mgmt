# perception/motion_analysis.py

import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self, motion_threshold=500, window_size=5, global_motion_threshold=0.15):
        self.motion_threshold = motion_threshold
        self.prev_gray = None
        self.window_size = window_size  # Number of frames for rolling window
        self.score_history = []  # Rolling window of normalized scores
        self.normalized_score_history = []  # Rolling window of normalized scores
        self.global_motion_threshold = global_motion_threshold  # Threshold for global frame motion
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def check_global_motion(self, prev_frame, curr_frame):
        """
        Check if overall scene has excessive motion (camera shake, scene cut, etc.)
        
        Args:
            prev_frame, curr_frame: consecutive frames
        Returns:
            dict with global motion status and confidence reduction
        """
        if prev_frame is None or prev_frame.shape != curr_frame.shape:
            return {
                "has_global_motion": False,
                "global_motion_score": 0.0,
                "confidence_reduction": 1.0
            }
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow for entire frame
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude of motion vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get global motion score (mean magnitude across entire frame)
        global_motion_score = np.mean(mag)
        
        # Determine if global motion exceeds threshold
        has_global_motion = global_motion_score > self.global_motion_threshold
        
        # Reduce confidence if global motion is detected
        # More global motion = less confidence in local detections
        if has_global_motion:
            # Confidence reduction factor (0.0 = no confidence, 1.0 = full confidence)
            confidence_reduction = max(0.2, 1.0 - (global_motion_score / (self.global_motion_threshold * 3)))
        else:
            confidence_reduction = 1.0
        
        return {
            "has_global_motion": has_global_motion,
            "global_motion_score": float(global_motion_score),
            "confidence_reduction": float(confidence_reduction)
        }

    def estimate_camera_motion(self, prev_frame, curr_frame):
        """
        Estimate camera motion using sparse optical flow (Lucas-Kanade)
        
        Args:
            prev_frame, curr_frame: consecutive frames
        Returns:
            dict with average camera motion vectors (dx, dy)
        """
        if prev_frame is None or prev_frame.shape != curr_frame.shape:
            return {
                "avg_dx": 0.0,
                "avg_dy": 0.0,
                "avg_magnitude": 0.0,
                "num_features": 0
            }
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect good features to track (corners) in previous frame
        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )
        
        if corners is None or len(corners) == 0:
            return {
                "avg_dx": 0.0,
                "avg_dy": 0.0,
                "avg_magnitude": 0.0,
                "num_features": 0
            }
        
        # Calculate optical flow using Lucas-Kanade
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None, **self.lk_params
        )
        
        # Filter good features (status == 1)
        if status is None:
            return {
                "avg_dx": 0.0,
                "avg_dy": 0.0,
                "avg_magnitude": 0.0,
                "num_features": 0
            }
        
        good_features = status.ravel() == 1
        
        if not np.any(good_features):
            return {
                "avg_dx": 0.0,
                "avg_dy": 0.0,
                "avg_magnitude": 0.0,
                "num_features": 0
            }
        
        # Calculate motion vectors for good features
        motion_vectors = next_pts[good_features] - corners[good_features]
        
        # Calculate average motion (camera motion)
        avg_dx = np.mean(motion_vectors[:, 0, 0])
        avg_dy = np.mean(motion_vectors[:, 0, 1])
        avg_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
        
        return {
            "avg_dx": float(avg_dx),
            "avg_dy": float(avg_dy),
            "avg_magnitude": float(avg_magnitude),
            "num_features": int(np.sum(good_features))
        }

    def analyze_motion(self, prev_frame, curr_frame, bbox):
        """
        Analyzes motion using optical flow (dense motion vectors)
        Motion score is normalized by bounding box area for comparable results
        across different sized detections.
        
        Args:
            prev_frame, curr_frame: consecutive frames
            bbox: (x, y, w, h)
        Returns:
            dict with motion level and optical flow data
        """
        x, y, w, h = bbox

        prev_roi = prev_frame[y:y+h, x:x+w]
        curr_roi = curr_frame[y:y+h, x:x+w]

        if prev_roi.size == 0 or curr_roi.size == 0:
            return {
                "motion_level": "NONE", 
                "score": 0.0, 
                "normalized_score": 0.0,
                "smoothed_score": 0.0,
                "optical_flow": None,
                "bbox_area": 0,
                "history_length": len(self.normalized_score_history)
            }

        prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

        # Use Farneback optical flow for dense motion estimation
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude and angle of motion vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Raw motion score (mean magnitude of all motion vectors)
        motion_score = np.mean(mag)
        max_motion = np.max(mag)
        
        # Calculate bounding box area
        bbox_area = w * h
        
        # Normalize motion score by bounding box area
        # This makes motion detection comparable across different sized detections
        # Smaller detections will have higher normalized scores for same motion intensity
        normalized_score = motion_score * (bbox_area / (640 * 480))  # normalized to 640x480 frame

        # Add normalized score to rolling window for temporal smoothing
        self.normalized_score_history.append(normalized_score)
        if len(self.normalized_score_history) > self.window_size:
            self.normalized_score_history.pop(0)
        
        # Calculate smoothed score using rolling window average
        smoothed_score = np.mean(self.normalized_score_history)

        # Classify motion level based on smoothed optical flow magnitude
        if smoothed_score > self.motion_threshold / 50:
            level = "HIGH"
        elif smoothed_score > self.motion_threshold / 200:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "motion_level": level,
            "score": float(motion_score),
            "normalized_score": float(normalized_score),
            "smoothed_score": float(smoothed_score),
            "max_motion": float(max_motion),
            "optical_flow": flow,
            "bbox_area": bbox_area,
            "history_length": len(self.normalized_score_history),
            "is_global_motion": False,
            "confidence": 1.0
        }

    def analyze_motion_with_camera_compensation(self, prev_frame, curr_frame, bbox):
        """
        Analyzes motion with camera motion compensation.
        Uses sparse optical flow to estimate camera motion and subtracts it from ROI motion.
        
        Args:
            prev_frame, curr_frame: consecutive frames
            bbox: (x, y, w, h) for region of interest
        Returns:
            dict with motion analysis compensated for camera movement
        """
        # Get local motion analysis
        local_motion = self.analyze_motion(prev_frame, curr_frame, bbox)
        
        # Estimate camera motion using sparse optical flow
        camera_motion = self.estimate_camera_motion(prev_frame, curr_frame)
        
        x, y, w, h = bbox
        
        # Compensate for camera motion by subtracting it from raw motion score
        # Camera motion magnitude is subtracted from the local optical flow
        compensated_motion_score = max(0.0, local_motion['score'] - camera_motion['avg_magnitude'])
        
        # Recalculate normalized score with compensated motion
        bbox_area = w * h
        compensated_normalized_score = compensated_motion_score * (bbox_area / (640 * 480))
        
        # Add compensated score to rolling window for temporal smoothing
        self.normalized_score_history.append(compensated_normalized_score)
        if len(self.normalized_score_history) > self.window_size:
            self.normalized_score_history.pop(0)
        
        # Calculate smoothed score using rolling window average
        smoothed_score = np.mean(self.normalized_score_history)
        
        # Classify motion level based on compensated smoothed score
        if smoothed_score > self.motion_threshold / 50:
            level = "HIGH"
        elif smoothed_score > self.motion_threshold / 200:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        # Determine if camera motion was significant enough to affect confidence
        camera_motion_significant = camera_motion['avg_magnitude'] > (self.global_motion_threshold * 2)
        
        return {
            "motion_level": level,
            "score": float(local_motion['score']),
            "compensated_score": float(compensated_motion_score),
            "normalized_score": float(local_motion['normalized_score']),
            "compensated_normalized_score": float(compensated_normalized_score),
            "smoothed_score": float(smoothed_score),
            "max_motion": float(local_motion['max_motion']),
            "optical_flow": local_motion['optical_flow'],
            "bbox_area": bbox_area,
            "history_length": len(self.normalized_score_history),
            "camera_motion_magnitude": camera_motion['avg_magnitude'],
            "camera_motion_dx": camera_motion['avg_dx'],
            "camera_motion_dy": camera_motion['avg_dy'],
            "num_tracked_features": camera_motion['num_features'],
            "camera_motion_compensated": camera_motion_significant,
            "is_global_motion": False,
            "confidence": 1.0
        }
        """
        Analyzes motion with global frame motion detection.
        If overall scene motion is high, suppresses motion classification responsiveness.
        
        Args:
            prev_frame, curr_frame: consecutive frames
            bbox: (x, y, w, h) for region of interest
        Returns:
            dict with motion analysis and confidence scores
        """
        # Get local motion analysis
        local_motion = self.analyze_motion(prev_frame, curr_frame, bbox)
        
        # Check for global frame motion (camera shake, pan, etc.)
        global_motion_info = self.check_global_motion(prev_frame, curr_frame)
        
        # Apply confidence reduction based on global motion
        confidence = global_motion_info['confidence_reduction']
        has_global_motion = global_motion_info['has_global_motion']
        
        # If global motion is detected, suppress motion level classification
        if has_global_motion:
            # Downgrade motion level to account for unreliable detections
            original_level = local_motion['motion_level']
            if original_level == "HIGH":
                suppressed_level = "MEDIUM"
            elif original_level == "MEDIUM":
                suppressed_level = "LOW"
            else:
                suppressed_level = "LOW"
        else:
            suppressed_level = local_motion['motion_level']
        
        # Return combined analysis
        return {
            "motion_level": suppressed_level,
            "original_motion_level": local_motion['motion_level'],
            "score": local_motion['score'],
            "normalized_score": local_motion['normalized_score'],
            "smoothed_score": local_motion['smoothed_score'],
            "max_motion": local_motion['max_motion'],
            "optical_flow": local_motion['optical_flow'],
            "bbox_area": local_motion['bbox_area'],
            "history_length": local_motion['history_length'],
            "is_global_motion": has_global_motion,
            "global_motion_score": global_motion_info['global_motion_score'],
            "confidence": confidence,
            "confidence_level": "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.5 else "LOW"
        }

    def draw_motion_vectors(self, frame, flow, step=15, scale=2):
        """
        Visualizes optical flow vectors on frame
        
        Args:
            frame: input frame
            flow: optical flow result
            step: spacing between vectors
            scale: scale of arrows
        """
        h, w = frame.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
        
        # Extract flow values at sampled points
        fx, fy = flow[y, x].T
        fx = fx.T
        fy = fy.T
        
        # Draw the motion vectors
        frame_copy = frame.copy()
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                x1, y1 = x[i, j], y[i, j]
                x2, y2 = int(x1 + fx[i, j] * scale), int(y1 + fy[i, j] * scale)
                
                # Draw arrow
                cv2.arrowedLine(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
                cv2.circle(frame_copy, (x1, y1), 2, (0, 255, 0), -1)
        
        return frame_copy

    def get_history_stats(self):
        """
        Get statistics from the score history
        
        Returns:
            dict with min, max, mean, and std of smoothed scores
        """
        if not self.normalized_score_history:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0
            }
        
        history_array = np.array(self.normalized_score_history)
        return {
            "min": float(np.min(history_array)),
            "max": float(np.max(history_array)),
            "mean": float(np.mean(history_array)),
            "std": float(np.std(history_array))
        }
    
    def reset_history(self):
        """Reset the motion score history"""
        self.normalized_score_history = []


def run_live_motion_analysis(camera_id=0):
    """
    Live motion analysis from camera feed
    
    Args:
        camera_id: camera device ID (default: 0 for webcam)
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Initialize analyzer with window size for temporal smoothing
    analyzer = MotionAnalyzer(motion_threshold=500, window_size=5)
    prev_frame = None
    
    print("Motion Analysis - Live Feed with Temporal Smoothing")
    print(f"Smoothing Window Size: {analyzer.window_size} frames")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("Press 'r' to reset history")
    print("-" * 50)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            
            if prev_frame is not None:
                # Analyze motion for the entire frame
                h, w = frame.shape[:2]
                bbox = (0, 0, w, h)
                
                # Use camera motion compensation for more accurate analysis
                motion_result = analyzer.analyze_motion_with_camera_compensation(prev_frame, frame, bbox)
                
                # Visualize optical flow
                if motion_result['optical_flow'] is not None:
                    frame_with_flow = analyzer.draw_motion_vectors(frame, motion_result['optical_flow'])
                else:
                    frame_with_flow = frame
                
                # Determine display color based on confidence and camera motion
                if motion_result['camera_motion_compensated']:
                    color_main = (0, 165, 255)  # Orange - camera motion detected
                else:
                    color_main = (0, 255, 0)  # Green - no camera motion
                
                # Add text overlay
                cv2.putText(
                    frame_with_flow,
                    f"Motion Level: {motion_result['motion_level']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color_main,
                    2
                )
                
                # Show camera motion info
                if motion_result['camera_motion_compensated']:
                    cv2.putText(
                        frame_with_flow,
                        f"Camera Motion Detected: {motion_result['camera_motion_magnitude']:.4f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2
                    )
                
                cv2.putText(
                    frame_with_flow,
                    f"Raw Score: {motion_result['score']:.4f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 200, 0),
                    1
                )
                
                cv2.putText(
                    frame_with_flow,
                    f"Camera Compensated: {motion_result['compensated_score']:.4f}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (100, 149, 237),
                    2
                )
                
                cv2.putText(
                    frame_with_flow,
                    f"Smoothed: {motion_result['smoothed_score']:.4f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    1
                )
                
                cv2.putText(
                    frame_with_flow,
                    f"Camera Motion (dx,dy): ({motion_result['camera_motion_dx']:.2f}, {motion_result['camera_motion_dy']:.2f})",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 255),
                    1
                )
                
                cv2.putText(
                    frame_with_flow,
                    f"Tracked Features: {motion_result['num_tracked_features']}",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 255),
                    1
                )
                
                cv2.putText(
                    frame_with_flow,
                    f"Window: {motion_result['history_length']}/{analyzer.window_size}",
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 200, 200),
                    1
                )
                cv2.putText(
                    frame_with_flow,
                    f"Frame: {frame_count}",
                    (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )
            else:
                frame_with_flow = frame
            
            cv2.imshow("Motion Analysis - Live Feed", frame_with_flow)
            
            prev_frame = frame.copy()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('s'):
                filename = f"motion_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame_with_flow)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                analyzer.reset_history()
                print("History reset")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    run_live_motion_analysis()