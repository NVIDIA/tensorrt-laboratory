# Istio

## Install

```
helm template install/kubernetes/helm/istio --name istio --namespace istio-system > yais-istio.yml
```

Modify the sidecar injetor to require annotations;

```
# Source: istio/templates/sidecar-injector-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: istio-sidecar-injector
  namespace: istio-system
  labels:
    app: istio
    chart: istio-0.8.0
    release: istio
    heritage: Tiller
    istio: sidecar-injector
data:
  config: |-
    policy: disabled  # <== default : enabled
    template: |-
...
```

Install Istio and enable the default namespace for injection; however, only
pods with the proper annotations will have sidecars injected.

```
kubectl apply -f yais-istio.yml
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
