#!/bin/bash

mkdir -p $HOME/.kube
touch $HOME/.kube/config

export MINIKUBE_HOME=$HOME
export CHANGE_MINIKUBE_NONE_USER=true
export KUBECONFIG=$HOME/.kube/config

version=v1.10

sudo minikube start \
  --feature-gates=DevicePlugins=true \
  --vm-driver=none \
  --kubernetes-version=${version}.0 \
  --bootstrapper=kubeadm \
  --extra-config=kubelet.authentication-token-webhook=true \
  --extra-config=kubelet.authorization-mode=Webhook \
  --extra-config=scheduler.address=0.0.0.0 \
  --extra-config=controller-manager.address=0.0.0.0 \
  --extra-config=controller-manager.cluster-signing-cert-file="/var/lib/localkube/certs/ca.crt" \
  --extra-config=controller-manager.cluster-signing-key-file="/var/lib/localkube/certs/ca.key" \
  --extra-config=apiserver.admission-control="NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota" 

if [ ! -e $HOME/.kube ]; then
  sudo mv /root/.kube $HOME/.kube > /dev/null 2>&1 ||: # this will write over any previous configuration
  sudo chown -R $USER $HOME/.kube > /dev/null 2>&1 ||:
  sudo chgrp -R $USER $HOME/.kube > /dev/null 2>&1 ||:
fi

if [ ! -e $HOME/.minikube ]; then
  sudo mv /root/.minikube $HOME/.minikube # > /dev/null 2>&1 ||: this will write over any previous configuration
  sudo chown -R $USER $HOME/.minikube > /dev/null 2>&1 ||:
  sudo chgrp -R $USER $HOME/.minikube > /dev/null 2>&1 ||:
fi

kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${version}/nvidia-device-plugin.yml

# dns fix for dgx-stations using ubuntu's network manager
kubectl apply -f https://raw.githubusercontent.com/ryanolson/k8s-upstream-dns/master/dns.yml
