from filterpy.kalman import KalmanFilter
import numpy as np
from config.settings import CONFIG

class Track:
    count = 0

    def __init__(self, bbox):
        Track.count += 1
        self.id = Track.count
        self.bbox = bbox
        self.time_since_update = 0
        self.hits = 1
        self.kf = self._create_kalman_filter(bbox)

    def _create_kalman_filter(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([cx, cy, 0, 0])
        kf.P *= 10
        kf.R *= 1
        return kf

class SortTracker:
    """
    SORT (Simple Online Realtime Tracking) tracker.
    Maintains identities for detected objects across frames.
    """

    def __init__(self):
        cfg = CONFIG.get("tracking")
        self.max_age = cfg["max_age"]
        self.min_hits = cfg["min_hits"]
        self.iou_thres = cfg["iou_threshold"]

        self.tracks = []

    def update(self, detections):
        updated_tracks = []

        # IOU matching (simplified)
        for det in detections:
            matched = False
            for track in self.tracks:
                iou = self._compute_iou(det["box"], track.bbox)
                if iou > self.iou_thres:
                    track.bbox = det["box"]
                    track.hits += 1
                    updated_tracks.append(track)
                    matched = True
                    break

            if not matched:
                new_track = Track(det["box"])
                updated_tracks.append(new_track)

        # Clean up dead tracks
        self.tracks = [t for t in updated_tracks if t.hits >= self.min_hits]
        return self.tracks

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0

        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

        return inter / float(areaA + areaB - inter)
