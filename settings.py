import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'salmon_detector_v1.pb')
MODEL_TRT_BIN_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'trt_ssd_mobilenet_v1.bin')
LIB_FLATTEN_PATH = os.path.join(CUR_DIR, 'utils', 'libflattenconcat.so')

CONFIDENCE = 0.2

LOCAL = False
