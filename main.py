"""
MVPS (Maritime Visual Perception System)
Safe, research-focused computer vision project.
"""

from core.detector import ObjectDetector
from core.tracker import SortTracker
from core.video_stream import VideoStream
from core.utils import draw_detections

import cv2

def main():
    print("[INFO] Initializing system...")

    detector = ObjectDetector()
    tracker = SortTracker()
    stream = VideoStream("sample_video.mp4")    # Replace with your video file

    print("[INFO] System ready. Processing video...")

    while True:
        ret, frame = stream.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        annotated = draw_detections(frame, detections, tracked_objects)

        cv2.imshow("MVPS - Research Edition", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    stream.release()
    print("[INFO] Processing finished.")

if __name__ == "__main__":
    main()
