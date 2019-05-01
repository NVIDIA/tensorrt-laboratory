#!/bin/bash

if [ ! -e "resnet50.tar.gz" ]; then
  wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
fi

if [ ! -e "open_source_images.tar.gz" ]; then
  wget https://s3-us-west-2.amazonaws.com/com.nvidia.tensorrt-laboratory/open_source_images.tar.gz
fi

if md5sum -c resnet50.md5; then
  if [ ! -e "resnet50" ]; then
    tar xzf resnet50.tar.gz
  fi
  echo "ResNet50 download good"
else
  echo "ResNet50 md5 checksum failed"
  exit 911
fi

if md5sum -c open_source_images.md5; then
  if [ ! -e "calibration_images" ]; then
    tar xf open_source_images.tar.gz
  fi
  echo "All good - Continue to Build Phase"
else
  echo "calibration_images md5 checksum failed"
  exit 911
fi
