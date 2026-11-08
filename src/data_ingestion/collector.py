import cv2
from pathlib import Path
from ultralytics import YOLO

DATA_DIR = Path("data/collected")

PERIODIC_DIR = DATA_DIR / "periodic"
CANDIDATE_DIR = DATA_DIR / "candidates"

SAMPLE_INTERVAL = 30       
MIN_CONFIDENCE = 0.10
LOW_CONFIDENCE_MAX = 0.35
MODEL_PATH = "yolo_26n.pt"

CAMERA_ID = 0


def main():
    model = YOLO(MODEL_PATH) #model

    camera = cv2.VideoCapture( #camera config for using webcam on laptop(i don't have ip cam :( )
        CAMERA_ID,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam")

    try:
        while True:
            ret, frame = camera.read()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()