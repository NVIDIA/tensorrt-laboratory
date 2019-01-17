#!/usr/bin/env python3
import os
import time

import numpy as np

import infer
import infer_test_utils as utils


def main():
    models = infer.InferenceManager(max_exec_concurrency=2)
    mnist = models.register_tensorrt_engine("mnist", "/work/models/onnx/mnist-v1.3/mnist-v1.3.engine")
    models.update_resources()

    print("Input Bindings: {}".format(mnist.input_bindings()))
    print("Output Bindings: {}".format(mnist.output_bindings()))

    inputs = utils.load_inputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    expected = utils.load_outputs("/work/models/onnx/mnist-v1.3/test_data_set_0")

    start = time.process_time()
    results = [mnist.infer(Input3=input) for input in inputs]
    results = [r.get() for r in results]
    print("Compute Time: {}".format(time.process_time() - start))

    for r, e in zip(results, expected):
        for key, val in r.items():
            print("Output Binding Name: {}; shape{}".format(key, val.shape))
            r = val.reshape((1,10))
            np.testing.assert_almost_equal(r, e, decimal=3) 

    models.serve()
    #mnist_model = models.get_model("mnist")
    #benchmark = infer.InferBench(models)
    #benchmark.run(mnist_model, 1, 0.1)
    #print(results)
 

if __name__ == "__main__":
    main()
