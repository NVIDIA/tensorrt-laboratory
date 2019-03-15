# TensorRT ResNet50 Example

- `fetch.sh` downloads the onnx model and test data from S3
  - after running this script the `resnet50` directory should be present in your
    local path

- `build.py` generates TensorRT engines from the `model.onnx` file
  - cli options:
    - `--batch` will select the batch size, multiple can be given, a separate
      engine for each batch size will be generated .e.g.
      ```
      ./buid.py --batch=1 --batch=8 resnet50/model.onnx
      ```
    - `--precision` can be `fp32`, `fp16`.  if multiple precision are given, an
      engine for each will be created.  the following commmand will build 4
      engines
      ```
      ./build.py --batch=1 --batch=8 --precision=fp16 --precision=fp32 resnet50/model.onnx
      ```

- `bench.py`


install: scikit-image-0.14.2 scipy-1.2.1

## Credits

 - [broccoli image](https://www.openfotos.com/view/broccoli-3784) - OpenFotosa
   - https://www.openfotos.com/pages/open-fotos-license
