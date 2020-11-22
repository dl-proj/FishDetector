import os
import ctypes

import uff
import tensorrt as trt
import graphsurgeon as gs

# from config import model_ssd_inception_v2_coco_2017_11_17 as model
from utils.nano_config import ssd_mobile_net_v1 as model
# from utils.nano_config import ssd_mobile_net_v2 as model
from settings import MODEL_PATH, MODEL_TRT_BIN_PATH, LIB_FLATTEN_PATH


def create_trt_model_bin():
    ctypes.CDLL(LIB_FLATTEN_PATH)

    # initialize
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')

    # compile model into TensorRT
    if not os.path.isfile(MODEL_TRT_BIN_PATH):
        dynamic_graph = model.add_plugin(gs.DynamicGraph(MODEL_PATH))
        uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.output_name, output_filename='tmp.uff')

        with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = 1 << 28
            builder.max_batch_size = 1
            builder.fp16_mode = True

            parser.register_input('Input', model.dims)
            parser.register_output('MarkOutput_0')
            parser.parse('tmp.uff', network)
            engine = builder.build_cuda_engine(network)

            buf = engine.serialize()
            with open(MODEL_TRT_BIN_PATH, 'wb') as f:
                f.write(buf)


if __name__ == '__main__':
    create_trt_model_bin()
