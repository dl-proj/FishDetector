import cv2
import os

from settings import LOCAL, MODEL_TRT_BIN_PATH, CUR_DIR


if __name__ == '__main__':
    if LOCAL:
        from src.pc.detector import SalmonDetector

        SalmonDetector().detect_from_images(frame=cv2.imread(os.path.join(CUR_DIR, 'salmon0.jpg')))
    else:
        from src.nano.trt_creator import create_trt_model_bin
        from src.nano.detector import SalmonDetectorNano

        if not os.path.exists(MODEL_TRT_BIN_PATH):
            create_trt_model_bin()
        SalmonDetectorNano().detect_salmon_frame(frame=cv2.imread(os.path.join(CUR_DIR, 'salmon0.jpg')))
