#!/usr/bin/env python3
import time
from concurrent import futures

import numpy as np
import py_yais as yais


def main(data):
    # Create TensorRT Inference Manager
    models = yais.InferenceManager(8)
    flowers50 = models.register_tensorrt_engine("/work/models/ResNet-50-b8-int8.engine", "flowers")
    flowers152 = models.register_tensorrt_engine("/work/models/ResNet-152-b8-int8.engine", "big-flowers")
    time.sleep(1)
    engines = models.cuda()

    # infer function
    def infer(input):
        batch_id, data = input
        future50 = flowers50.pyinfer(data)
        #future152 = engines.get_model("big-flowers").pyinfer(data)
        return (batch_id, future50)

    # benchmark performance
    start = time.time()
    with futures.ThreadPoolExecutor(max_workers=20) as pool:
        results = pool.map(infer, enumerate(data))
    for batch_id, f50 in results:
        f50.get()
        print("batch_id {} finished".format(batch_id))
    end = time.time()
    print("Finished {} batches in {}; {} inf/sec".format(
        len(data), end-start, len(data)*data[0].shape[0] / (end-start)))

    flowers50(data=np.random.random_sample(size=(2,2)))

    #engines.serve()

if __name__ == "__main__":
    print("Generating Random Data")
    data = [np.random.random_sample(size=(8,3,224,224)) for _ in range(10)]
    print("Starting Inference Loop")
    main(data)
