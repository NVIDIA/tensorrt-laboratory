#!/bin/bash

default_engine=/work/models/ResNet-152-b8-fp16.engine
concurrency=${YAIS_CONCURRENCY:-1}
engine=${YAIS_TRT_ENGINE:-$default_engine}

if [ "$engine" = "$default_engine" ]; then
  if [ ! -e $engine ]; then
    cd /work/models
    ./setup.py
  fi
fi

/work/build/examples/02_TensorRT_GRPC/inference-grpc.x \
    --engine=${engine} \
    --contexts=${concurrency}
