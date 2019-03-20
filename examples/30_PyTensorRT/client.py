#!/usr/bin/env python3

## Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##  * Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
##  * Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  * Neither the name of NVIDIA CORPORATION nor the names of its
##    contributors may be used to endorse or promote products derived
##    from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
## OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import time

import numpy as np

import infer
import infer_test_utils as utils

def main():
    manager = infer.RemoteInferenceManager(hostname="localhost:50052")
    models = manager.get_models()
    print(models)

    mnist = manager.infer_runner("mnist")

    print("Input Bindings: {}".format(mnist.input_bindings()))
    print("Output Bindings: {}".format(mnist.output_bindings()))

    inputs = utils.load_inputs("/work/models/onnx/mnist-v1.3/test_data_set_0")
    expected = utils.load_outputs("/work/models/onnx/mnist-v1.3/test_data_set_0")

    start = time.process_time()
    results = [mnist.infer(Input3=input) for input in inputs]
    results = [r.get() for r in results]
    print("Compute Time: {}".format(time.process_time() - start))
    print(results)

#   for r, e in zip(results, expected):
#       for key, val in r.items():
#           print("Output Binding Name: {}; shape{}".format(key, val.shape))
#           r = val.reshape((1,10))
#           np.testing.assert_almost_equal(r, e, decimal=3) 

#   models.serve()
    #mnist_model = models.get_model("mnist")
    #benchmark = infer.InferBench(models)
    #benchmark.run(mnist_model, 1, 0.1)
    #print(results)
 

if __name__ == "__main__":
    main()
