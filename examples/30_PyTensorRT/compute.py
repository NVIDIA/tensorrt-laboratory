#!/usr/bin/env python3
import time
from concurrent import futures

import numpy as np
import infer as yais

import onnx
import os
import glob

from onnx import numpy_helper

path = "/work/models/onnx/mnist-v1.3"
test = "test_data_set_0"
test_data_dir = os.path.join(path, test)

def load_inputs(test_data_dir):
    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))
    return inputs

def load_outputs(test_data_dir):
    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(tensor))
    return ref_outputs

def infer(data):
    # Create TensorRT Inference Manager
    models = yais.InferenceManager(1)
    #flowers50 = models.register_tensorrt_engine("flowers", "/work/models/ResNet-50-b1-fp32.engine")
    mnist = models.register_tensorrt_engine("mnist", "/work/models/onnx/mnist-v1.3/mnist-v1.3.engine")
    engines = models.cuda()

    # benchmark performance
    # time.sleep(2)
    future = mnist.infer(data)
    start = time.time()
    print("queued")
    future.wait()
    result = future.get()
    print(result)
    print(result.shape)
    print("finished")
    end = time.time()
    print(end-start)
    return result.reshape((1,10))

if __name__ == "__main__":
    inputs = load_inputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    outputs = load_outputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    results = [infer(inputs)]
    print(results)
    print(outputs)
    np.testing.assert_almost_equal(results, outputs, decimal=2) 
