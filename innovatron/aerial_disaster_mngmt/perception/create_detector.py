code = '''# perception/human_detection.py

import cv2
import numpy as np

class HumanDetector:
    def __init__(self, model_path=None):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.focal_length = 800
        self.known_height = 170
        self.min_height = 30
        self.min_width = 20
        self.max_height_ratio = 3.5
        self.min_height_ratio = 1.2

    def calculate_distance(self, bbox_height):
        if bbox_height == 0:
            return float('inf')
        distance_cm = (self.focal_length * self.known_height) / bbox_height
        return distance_cm

    def is_valid_human_detection(self, x, y, w, h):
        if w < self.min_width or h < self.min_height:
            return False
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < self.min_height_ratio or aspect_ratio > self.max_height_ratio:
            return False
        return True

    def detect_humans(self, frame):
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
                "label": f"Human #{detection_id} ({distance_m:.2f}m)"
            })
            detection_id += 1
        return detections

    def _group_detections(self, rects):
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


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        exit(1)
    detector = HumanDetector()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect_humans(frame)
        frame_count += 1
        info_text = f"Frame: {frame_count} | Detections: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: {len(detections)} humans detected")
        for d in detections:
            x, y, w, h = d["bbox"]
            distance_m = d["distance_m"]
            label = d["label"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label_text = f"{label} - {distance_m:.2f}m away"
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
'''

with open("human_detection.py", "w") as f:
    f.write(code)
print("File created")
