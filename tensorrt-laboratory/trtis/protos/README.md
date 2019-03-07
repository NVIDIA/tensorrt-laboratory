# TRTIS Protos

I have to manually copy all the .protos from the [TensorRT Inference Server](https://github.com/nvidia/tensorrt-inference-server)
repo.

Unfortunately, there is now good way to link again TRTIS as an external library.

Tracking:
https://github.com/NVIDIA/tensorrt-inference-server/issues/110


I also manually removed the absolute include paths that are unnecessary and a
weird vestiage of the TRTIS project directory.