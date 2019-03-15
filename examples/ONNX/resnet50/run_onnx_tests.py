import os

import trtlab
import numpy as np

import click
import onnx_utils as utils

tests = {}

def tensorrt_init(engines):
    manager = trtlab.InferenceManager(max_exec_concurrency=4)
    runners = []
    for engine in engines:
        name, _ = os.path.splitext(os.path.basename(engine))
        runners.append(manager.register_tensorrt_engine(name, engine))
    manager.update_resources()
    return runners

def test_data(test_path):
    for path, dirs, files in os.walk(test_path):
        if os.path.basename(path).startswith("test_"):
            tests[path] = files 
    for path, files in tests.items():
        inputs = utils.load_inputs(path)
        outputs = utils.load_outputs(path)
        print("** Testing {} **".format(path))
        yield inputs, outputs

def run_test(runner, inputs, outputs):
    inputs = preprocess_inputs(runner, inputs)
    future = runner.infer(**inputs)
    result = future.get()
    validate_results(result, outputs)

def preprocess_inputs(runner, inputs):
    expected_input = runner.input_bindings()
    if len(expected_input) != len(inputs):
        raise RuntimeError("mismatched number of inputs")
    keys = list(expected_input.keys())
    input_name = keys[0]
    info = expected_input[keys[0]]
    shape = info['shape']
    tensor = inputs[0]
    batch_size = tensor.shape[0]
    if list(shape) != list(tensor.shape[1:]):
        raise RuntimeError("mismatched input dimensions")
    return { input_name: tensor }

def validate_results(computed, expected):
    keys = list(computed.keys())
    output_name = keys[0]
    output_value = computed[output_name]
    print(output_value.shape)
    print(expected[0].shape)
    np.testing.assert_almost_equal(output_value, expected[0], decimal=3)
    print("-- Test Passed: All outputs {} match within 3 decimals".format(output_value.shape))

File = click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
Path = click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)

@click.command()
@click.option("--tests", type=Path, default="resnet50")
@click.argument("engine", type=File, nargs=1)
def main(engine, tests):
    runners = tensorrt_init([engine])
    for runner in runners:
        for inputs, outputs in test_data(tests):
            run_test(runner, inputs, outputs)
    

if __name__ == "__main__":
    main() 
