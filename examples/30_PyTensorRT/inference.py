#!/usr/bin/env python3

import py_yais as yais
#import numpy as np

#def random_data(*, count=1000, shape=(3,224,224)):
#    for _ in range(count):
#        yield np.random.uniform(size=shape)

def flowers():
    manager = yais.InferenceManager(4)
    manager.register_tensorrt_engine("/work/models/ResNet-50-b8-int8.engine", "flowers")

if __name__ == "__main__":
    flowers()
   
#    with futures.ThreadPoolExecutor(max_workers=100) as pool:
#        results = pool.map(flowers.infer, random_data(count=1000))
