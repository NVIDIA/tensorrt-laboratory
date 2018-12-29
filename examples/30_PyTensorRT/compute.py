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

def infer(model, data):
    # benchmark performance
    # time.sleep(2)
    #future = mnist.infer(data)
    #future = mnist.test(Input3=data)
    future = model.infer(Input3=data)
    start = time.time()
    print("queued")
    result = future.get()
    print("finished")
    end = time.time()
    print(end-start)
    return result

if __name__ == "__main__":
    models = yais.InferenceManager(1)
    mnist = models.register_tensorrt_engine("mnist", "/work/models/onnx/mnist-v1.3/mnist-v1.3.engine")
    models.cuda()

    inputs = load_inputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    outputs = load_outputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    results = infer(mnist, inputs[0])
    for key, val in results.items():
        print(key)
        results = [ val.reshape((1,10)) ]
        np.testing.assert_almost_equal(results, outputs, decimal=2) 
