import cv2
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime


DATA_DIR = Path("data/collected")

PERIODIC_DIR = DATA_DIR / "periodic"
CANDIDATE_DIR = DATA_DIR / "candidates"

SAMPLE_INTERVAL = 30       
MIN_CONFIDENCE = 0.10
LOW_CONFIDENCE_MAX = 0.35
MODEL_PATH = "yolo26n.pt"

CAMERA_ID = 0

def save_frame(frame, collection_type):
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")


    if(collection_type == "periodic"):
        save_dir = PERIODIC_DIR / date_folder

    save_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{collection_type}_{timestamp}.jpg"
    
    image_path = save_dir / file_name

    success = cv2.imwrite(str(image_path), frame)

    if not success:
        print(f"[ERROR] Cannot save: {image_path}")
        return

    print(f"[SAVED] {collection_type}: {image_path}")

    


def main():
    model = YOLO(MODEL_PATH) #model

    camera = cv2.VideoCapture( #camera config for using webcam on laptop(i don't have ip cam :( )
        CAMERA_ID,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        raise RuntimeError("[ERROR] Cannot open webcam")

    last_periodic_save = time.monotonic()
    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("[ERROR] Cannot read frame")
                continue

            current_time = time.monotonic()
            
            if(current_time - last_periodic_save >= SAMPLE_INTERVAL):
                save_frame(frame,"periodic")
                last_periodic_save = current_time
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()