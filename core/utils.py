import cv2
from config.settings import CONFIG

def draw_detections(frame, detections, tracks):
    draw_labels = CONFIG.get("display")["draw_labels"]
    draw_boxes = CONFIG.get("display")["draw_boxes"]

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        color = (0, 255, 0)

        if draw_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if draw_labels:
            cv2.putText(
                frame,
                f"ID {track.id}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
    return frame
