#!/bin/bash

concurrency=${YAIS_CONCURRENCY:-1}

if [ ! -e /work/models/ResNet-50-b1-fp32.engine ]; then
  cd /work/models
  ./setup.py
fi

/work/build/examples/02_TensorRT_GRPC/inference-grpc.x \
    --engine=/work/models/ResNet-50-b1-int8.engine \
    --contexts=${concurrency}
