#!/usr/bin/env python3
import time
from concurrent import futures

import numpy as np
import infer

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


if __name__ == "__main__":
    models = infer.InferenceManager(max_executions=2)
    mnist = models.register_tensorrt_engine("mnist", "/work/models/onnx/mnist-v1.3/mnist-v1.3.engine")
    print(mnist.input_bindings())
    models.update_resources()

    inputs = load_inputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    outputs = load_outputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    start = time.process_time()
    results = [mnist.infer(Input3=input) for input in inputs]
    results = [r.get() for r in results]
    print("Compute Time: {}".format(time.process_time() - start))
    for r in results:
        for key, val in r.items():
            print("Output Binding Name: {}; shape{}".format(key, val.shape))
            result = [ val.reshape((1,10)) ]
            np.testing.assert_almost_equal(result, outputs, decimal=3) 
