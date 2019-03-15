#!/bin/bash

if [ ! -e "resnet50.tar.gz" ]; then
  wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
fi

if md5sum -c resnet50.md5; then
  if [ ! -e "resnet50" ]; then
    tar xzf resnet50.tar.gz
  fi
  echo "All good - Continue to Build Phase"
else
  echo "md5 checksum failed"
  exit 911
fi

