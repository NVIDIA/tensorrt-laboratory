#!/bin/bash

if ! [ -x "$(command -v helm)" ]; then
  echo 'Error: helm is not installed.' >&2
  exit 1
fi

kubectl create -f service-account.yml

helm init --wait --service-account tiller

helm install coreos/prometheus-operator \
  --name prometheus-operator \
  --namespace monitoring

helm install coreos/kube-prometheus \
  --name kube-prometheus \
  --namespace monitoring \
  -f custom-settings.yml

kubectl apply -f nv-inference-prom.yml
