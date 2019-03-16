# TensorRT ResNet50 Example

- `fetch.sh` downloads the onnx model and test data from S3
  - after running this script the `resnet50` directory should be present in your
    local path

- Build (`build.py`) TensorRT engines from the `model.onnx` file
  - cli options:
    - `--batch` will select the batch size, multiple can be given, a separate
      engine for each batch size will be generated.
    - `--precision` can be `fp32`, `fp16`.  if multiple precision are given, an
      engine for each will be created.  
  - If you have a Turing or Volta GPU, then run the following commmand which will generates 4 engines:
    ```
    ./build.py --batch=1 --batch=8 --precision=fp16 --precision=fp32 resnet50/model.onnx
    ```
  - If you have a Pascal GPU, run the following which generates 2 engines:
    ```
    ./build.py --batch=1 --batch=8 --precision=fp32 resnet50/model.onnx
    ```
- Functional Test
  - `./run_onnx_tests.py model-b1-fp16.engine` will run the onnx tests 

- Benchmark TensorRT engines at different batch sizes and concurrent executions:
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b1-fp16.engine --contexts=1`
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b1-fp16.engine --contexts=8`
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b8-fp16.engine --contexts=1`
  - `/work/build/examples/00_TensorRT/infer.x --engine=model-b8-fp16.engine --contexts=6`

- `./run_jpeg_test.py --image=images/broccoli-3784.jpg model-b1-fp16.engine`
  - On a V100 using FP16, your results should be
  ```
  *** Results ***
  broccoli 0.9511453
  ```

## Credits

 - [broccoli image](https://www.openfotos.com/view/broccoli-3784) - OpenFotosa
   - https://www.openfotos.com/pages/open-fotos-license

## TODOs

 - [ ] Add Int8 calibration example
 - [ ] Update `run_jpeg_test.py` to highlight the async interface.
