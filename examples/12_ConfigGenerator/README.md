# TensorRT Inference Server Model Store Builder

- Ensure you built the project.
- Run `./link.sh` in this directory

## Design Requirements

This example consists of a ModelStore manager (Python) and a
ModelConfigGenerator (C++ w/ Python bindings).

The ModelConfigGenerator shall:
  - [X] parse serialized TensorRT engine files
  - [X] translate the necessary properties of the ICudaEngine to an
    `::nvidia::inferenceserver::ModelConfig` protobuf message
  - [ ] not require the presence of Cuda or a GPU to perform the actions

The ModelStore manager consists of a Python class for direct consumption and a
command-line application that shall:
  - [ ] create and manage a model-store in a user-supplied filesystem directory 
  - [X] add TensorRT model files to the model store using the
    ModelConfigGenerator and user-specified arguments
  - [ ] add new version of TensorRT models to a ModelStore
  - [ ] remove versions of entire models from the ModelStore
  - [ ] add, edit, update and remove Tensorflow models
  - [ ] add, edit, update and remove PyTorch/Caffe2 models

## Prototype Implementation

```
./ms_mgmt --help
Usage: ms_mgmt [OPTIONS]

Options:
  --engine PATH          TensorRT serialized engine  [required]
  --concurrency INTEGER  max number of concurrency executions allowed
  --name TEXT            model name; default to basename(engine) with the ext
                         dropped
  --version INTEGER      model version
  --store-path TEXT      model store path; default to ./model-store
  --help                 Show this message and exit.
```

```
./ms_mgmt --store-path=/tmp/model-store --engine=/work/models/ResNet-50-b1-fp32.engine --name=overridden-model-name --version=1337 --concurrency=10

ls /tmp/model-store/
overridden-model-name

ls /tmp/model-store/overridden-model-name/1337/
ResNet-50-b1-fp32.engine  model.plan

cat /tmp/model-store/overridden-model-name/config.pbtxt
name: "overridden-model-name"
platform: "tensorrt_plan"
max_batch_size: 1
input {
  name: "data"
  data_type: TYPE_FP32
  dims: 3
  dims: 224
  dims: 224
}
output {
  name: "prob"
  data_type: TYPE_FP32
  dims: 1000
  dims: 1
  dims: 1
}
instance_group {
  count: 10
  gpus: 0
}
```
