import cv2

class VideoStream:
    """
    Simple wrapper for OpenCV video capture & display.
    Handles:
    - Video input
    - Frame iteration
    - Window management
    """

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise Exception(f"[ERROR] Failed to open video source: {source}")

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
