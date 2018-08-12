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
kubectl create namespace istio-system
kubectl apply -f istio/istio-v1.0-minikube.yml

sleep 30
kubectl label namespace default istio-injection=enabled

# deploy yais example
kubectl apply -f yais-deploy.yml

# sleep 15
# kubectl label namespace default istio-injection-
