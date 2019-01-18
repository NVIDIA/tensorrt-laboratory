#!/usr/bin/env python3
import itertools
import os
import time

import numpy as np

import infer
import infer_test_utils as utils


def main():
    models = infer.InferenceManager(max_exec_concurrency=1)
    mnist = models.register_tensorrt_engine("mnist", "/work/models/onnx/mnist-v1.3/mnist-v1.3.engine")
    models.update_resources()

    print("Input Bindings: {}".format(mnist.input_bindings()))
    print("Output Bindings: {}".format(mnist.output_bindings()))

    inputs = utils.load_inputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    expected = utils.load_outputs("/work/models/onnx/mnist-v1.3/test_data_set_0")

    start = time.process_time()
    while True:
        futures = [mnist.infer(Input3=inputs[0]) for _ in range(100)]
        results = [f.get() for f in futures]
#   while True:
#       results = [mnist.infer(Input3=input) for input in itertools.repeat(inputs[0], 1000)]
#       results = [r.get() for r in results]
#       time.sleep(0.1)
    print("Compute Time: {}".format(time.process_time() - start))
    
#   for r, e in zip(results, expected):
#       for key, val in r.items():
#           print("Output Binding Name: {}; shape{}".format(key, val.shape))
#           r = val.reshape((1,10))
#           np.testing.assert_almost_equal(r, e, decimal=3) 

    #mnist_model = models.get_model("mnist")
    #benchmark = infer.InferBench(models)
    #benchmark.run(mnist_model, 1, 0.1)
    #print(results)
 

if __name__ == "__main__":
    main()
