import cv2
import time
import numpy as np
import pycuda.driver as cuda

import tensorrt as trt
from utils.nano_config import ssd_mobile_net_v1 as model
# from utils.nano_config import ssd_mobile_net_v2 as model
from settings import MODEL_TRT_BIN_PATH


class SalmonDetectorNano:
    def __init__(self):
        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)
        with open(MODEL_TRT_BIN_PATH, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)

        # create buffer
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        self.context = engine.create_execution_context()

    def detect_salmon_frame(self, frame):

        salmons = []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (model.dims[2], model.dims[1]))
        image = (2.0 / 255.0) * image - 1.0
        image = image.transpose((2, 0, 1))
        np.copyto(self.host_inputs[0], image.ravel())

        start_time = time.time()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        print("execute times " + str(time.time() - start_time))

        output = self.host_outputs[0]
        height, width, channels = frame.shape
        for i in range(int(len(output) / model.layout)):
            prefix = i * model.layout
            conf = output[prefix + 2]
            x_min = int(output[prefix + 3] * width)
            y_min = int(output[prefix + 4] * height)
            x_max = int(output[prefix + 5] * width)
            y_max = int(output[prefix + 6] * height)
            salmons.append([x_min, y_min, x_max, y_max])

            if conf > 0.7:
                print("Detected Salmon with confidence {}".format("{0:.0%}".format(conf)))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                cv2.putText(frame, "Salmon", (x_min + 10, y_min + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

        # cv2.imwrite("result.jpg", frame)
        # cv2.imshow("result", frame)
        # cv2.waitKey(0)
        return salmons


if __name__ == '__main__':
    SalmonDetectorNano().detect_salmon_frame(frame_path="")
