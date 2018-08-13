#!/bin/bash

kubectl create namespace istio-system
kubectl apply -f istio-v1.0-minikube.yml

# kubectl label namespace default istio-injection=enabled
# kubectl label namespace default istio-injection-
