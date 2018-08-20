#!/bin/bash

kubectl create -f service-account.yml

helm init --wait --service-account tiller

helm repo add coreos https://s3-eu-west-1.amazonaws.com/coreos-charts/stable/

helm install coreos/prometheus-operator \
  --name prometheus-operator \
  --namespace monitoring

helm install coreos/kube-prometheus \
  --name kube-prometheus \
  --namespace monitoring \
  -f custom-settings.yml

kubectl apply -f yais-metrics.yml
