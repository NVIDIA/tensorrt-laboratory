# TensorRT ResNet50 Example

- `fetch.sh` downloads the onnx model and test data from S3
  - after running this script the `resnet50` directory should be present in your
    local path

- `build.py` generates TensorRT engines from the `model.onnx` file
  - cli options:
    - `--batch` will select the batch size, multiple can be given, a separate
      engine for each batch size will be generated.
    - `--precision` can be `fp32`, `fp16`.  if multiple precision are given, an
      engine for each will be created.  the following commmand will build 4
      engines
    - run the following:
      ```
      ./build.py --batch=1 --batch=8 --precision=fp16 resnet50/model.onnx
      ```
- `python3 run_onnx_tests.py model-b1-fp16.engine` will run the onnx tests 

- benchmark engines at different batch sizes and concurrent executions:
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b1-fp16.engine --contexts=1`
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b1-fp16.engine --contexts=8`
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b8-fp16.engine --contexts=1`
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b8-fp16.engine --contexts=6`


## Credits

 - [broccoli image](https://www.openfotos.com/view/broccoli-3784) - OpenFotosa
   - https://www.openfotos.com/pages/open-fotos-license
