#!/bin/bash
cd /work/build/tensorrt-laboratory/python
make -j
cd /work/examples/30_PyTensorRT
if [ ! -e infer.cpython-35m-x86_64-linux-gnu.so ]; then
  ln -s /work/build/tensorrt-laboratory/python/tensorrt/infer.cpython-35m-x86_64-linux-gnu.so
fi
