# Istio

## Install

```
# Download the latest release
curl -L https://git.io/getLatestIstio | sh -

# Istio 1.0
helm template install/kubernetes/helm/istio --name istio --namespace istio-system \
  --set gateways.istio-ingressgateway.type=NodePort \
  --set gateways.istio-egressgateway.type=NodePort  > istio-v1.0-minikube.yml
```

Install Istio and enable the default namespace for injection; however, only
pods with the proper annotations will have sidecars injected.

```
kubectl create namespace istio-system
kubectl apply -f istio-v1.0-minikube.yml
kubectl label namespace default istio-injection=enabled
kubectl get namespace -L istio-injection
```

The annotation required for sidecar injection:
```
apiVersion: extensions/v1beta1
kind: Deployment
...
spec:
  template:
    metadata:
      annotations:                         # <== sidecar
        sidecar.istio.io/inject: "false"   # <== annotation
...
```
