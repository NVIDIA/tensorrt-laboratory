#!/usr/bin/env python3

import os
import time

import trtlab
import onnx_utils as utils

import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet.gluon.data.vision import transforms

from imagenet_labels import labels

import click

tests = {}

def tensorrt_init(engines):
    manager = trtlab.InferenceManager(max_exec_concurrency=4)
    runners = []
    for engine in engines:
        name, _ = os.path.splitext(os.path.basename(engine))
        runners.append(manager.register_tensorrt_engine(name, engine))
    manager.update_resources()
    return runners

def infer_image(runner, image):
    inputs = preprocess_image(runner, image)
    future = runner.infer(**inputs)
    result = future.get()
    for name, tensor in result.items():
        tensor = tensor.reshape(1000)
        idx = np.argmax(tensor) 
        print("\n*** Results ***") 
        print(labels[idx], tensor[idx])
        print("") 

def preprocess_image(runner, image_path):
    inputs = runner.input_bindings()
    keys = list(inputs.keys())
    input_name = keys[0]
    img = np.array(plt.imread(image_path))
    img = transform_image(img)
    return { input_name: img }

def transform_image(img):
    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(mx.nd.array(img)).asnumpy()
    img = np.expand_dims(img, axis=0) # batchify
    return img


def validate_results(computed, expected):
    keys = list(computed.keys())
    output_name = keys[0]
    output_value = computed[output_name]
    np.testing.assert_almost_equal(output_value, expected[0], decimal=3)
    print("-- Test Passed: All outputs {} match within 3 decimals".format(output_value.shape))

File = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
Path = click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)

@click.command()
@click.option("--image", type=File, multiple=True)
@click.argument("engine", type=File, nargs=1)
def main(engine, image):
    runners = tensorrt_init([engine])
    for runner in runners:
        for img in image:
            infer_image(runner, img) 
    

if __name__ == "__main__":
    main() 
