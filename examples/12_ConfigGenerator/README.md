# TensorRT Inference Server Model Store Builder

- Ensure you built the project.
- Run `./link.sh` in this directory

```
./trtis_config_gen --help
Usage: trtis_config_gen [OPTIONS]

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
./trtis_config_gen --store-path=/tmp/model-store --engine=/work/models/ResNet-50-b1-fp32.engine --name=overridden-model-name --version=1337 --concurrency=10

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
