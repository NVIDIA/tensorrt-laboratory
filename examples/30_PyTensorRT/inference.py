#!/usr/bin/env python3
import time
from concurrent import futures
import py_yais as yais
import numpy as np

def infer(model, input):
    return model.pyinfer(input).get()

def main(data):
    manager = yais.InferenceManager(8)
    flowers = manager.register_tensorrt_engine("/work/models/ResNet-50-b8-int8.engine", "flowers")
    time.sleep(1)
    manager.allocate_resources()
    def wrapped_infer(input):
        batch_id, data = input
        infer(flowers, data)
        print("finished batch_id={}".format(batch_id))
    start = time.time()
    with futures.ThreadPoolExecutor(max_workers=20) as pool:
        results = pool.map(wrapped_infer, enumerate(data))
    end = time.time()
    print("Finished {} batches in {}; {} inf/sec".format(
        len(data), end-start, len(data)*data[0].shape[0] / (end-start)))

if __name__ == "__main__":
    print("Generating Random Data")
    data = [np.random.random_sample(size=(8,3,224,224)) for _ in range(1000)]
    print("Starting Inference Loop")
    main(data)
   
#    with futures.ThreadPoolExecutor(max_workers=100) as pool:
#        results = pool.map(flowers.infer, random_data(count=1000))
