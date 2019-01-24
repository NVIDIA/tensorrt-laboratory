# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np
import ctypes

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "."))
import common

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

class ModelData(object):
    MODEL_PATH = "/work/models/flowers-152.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file, calibrator=None):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        builder.max_batch_size = 8
        precision = "fp32"
        if calibrator:
            builder.int8_mode = True
            builder.int8_calibrator = calibrator
            precision = "int8"
        else:
            builder.fp16_mode = True
            precision = "fp16"
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)
        serialized = engine.serialize()
        with open("/work/models/flowers-152-b{}-{}.engine".format(builder.max_batch_size, precision), "wb") as file:
            file.write(serialized)
        return engine

def normalize_image(image_name):
    image = Image.open(image_name)
    # Resize, antialias and transpose the image to CHW.
    c, h, w = ModelData.INPUT_SHAPE
    image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
    # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
    return ((image_arr / 255.0) - 0.5) * 2.0

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(test_image))
    return test_image

def create_calibration_dataset():
    jpegs = []
    for dirpath, subdirs, files in os.walk("/work/models/flowers-data/flowers"):
        for f in files:
            if f.endswith("jpg"):
                jpegs.append(os.path.join(dirpath, f))
    random.shuffle(jpegs)
    return jpegs[:200]

class ImageBatchStream:
    def __init__(self, batch_size, calibration_files):
        c, h, w = ModelData.INPUT_SHAPE
        self.batch_size = batch_size
        self.files = calibration_files
        self.batch = 0
        self.max_batches = (len(calibration_files) // batch_size) + \
                           (1 if (len(calibration_files) % batch_size) else 0)
        self.calibration_data = np.zeros((batch_size, c, h, w), dtype=np.float32)

    def reset(self):
        self.batch = 0
     
    def next_batch(self):
        c, h, w = ModelData.INPUT_SHAPE
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch : \
                                  self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                print("[ImageBatchStream] Processing ", f)
                img = normalize_image(f)
                imgs.append(img.reshape((c, h, w)))
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])


class MyEntropyCalibrator(trt.IInt8EntropyCalibrator):
    def __init__(self, stream):
        trt.IInt8EntropyCalibrator.__init__(self)
        self.batchstream = stream
        self.d_input = cuda.mem_alloc(self.batchstream.calibration_data.nbytes)
        stream.reset()

    def get_batch_size(self):
        return self.batchstream.batch_size

    def get_batch(self, bindings, names):
        batch = self.batchstream.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)
        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self, length):
        return None

    def write_calibration_cache(self, ptr, size):
#       cache = ctypes.c_char_p(int(ptr))
#       with open('calibration_cache.bin', 'wb') as f:
#           f.write(cache.value)
        return None



def main():
    calibration_files = create_calibration_dataset()
    batch_stream = ImageBatchStream(8, calibration_files)
    int8_calibrator = None
    int8_calibrator = MyEntropyCalibrator(batch_stream)
    engine = build_engine_onnx("/work/models/flowers-152.onnx", calibrator=int8_calibrator)
#   serialized = engine.serialize()
#   with open("/work/models/flowers-152-b8-int8.engine", "wb") as file:
#       file.write(serialized)
#   h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
#   with engine.create_execution_context() as context:
#       for test_image in ["/work/models/flowers-data/test/image_07927.jpg",
#                          "/work/models/flowers-data/test/image_06969.jpg",]:
#           #test_image = "/work/models/flowers-data/test/image_07927.jpg" # 13 - blanket flower
#           #test_image = "/work/models/flowers-data/test/image_06969.jpg"  # 0 - alpine sea holly
#           test_case = load_normalized_test_case(test_image, h_input)
#           do_inference(context, h_input, d_input, h_output, d_output, stream)
#           # We use the highest probability as our prediction. Its index corresponds to the predicted label.
#           pred = np.argmax(h_output)
#           score = softmax(h_output)[pred]
#           print("Recognized " + test_case + " as " + str(pred) + " score: " + str(score))

def old_main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    data_path, data_files = common.find_sample_data(description="Runs a ResNet50 network with a TensorRT inference engine.", subfolder="resnet50", find_files=["binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg", ModelData.MODEL_PATH, "class_labels.txt"])
    # Get test images, models and labels.
    test_images = data_files[0:3]
    onnx_model_file, labels_file = data_files[3:]
    labels = open(labels_file, 'r').read().split('\n')

    # Build a TensorRT engine.
    with build_engine_onnx(onnx_model_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            test_image = random.choice(test_images)
            test_case = load_normalized_test_case(test_image, h_input)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            pred = np.argmax(h_output)
            print("Recognized " + test_case + " as " + pred)



if __name__ == '__main__':
    main()
