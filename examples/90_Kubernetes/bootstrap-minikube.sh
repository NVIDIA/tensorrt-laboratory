#!/bin/bash

if ! [ -x "$(command -v helm)" ]; then
  echo 'Error: helm is not installed.' >&2
  exit 1
fi

# minikube
(cd minikube && ./bootstrap.sh)

# prometheus-operator
(cd prometheus && ./bootstrap.sh)

# istio
(cd istio && ./bootstrap.sh)
