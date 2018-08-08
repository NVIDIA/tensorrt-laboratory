# Kubernetes

Using [Kubernetes on NVIDIA GPUs, aka KONG](https://developer.nvidia.com/kubernetes-gpu) is a great
way of deploying GPU accelerated microservices.  This page will act as a guide for
for both development and production deployment.

* For development, we will use [minikube](https://kubernetes.io/docs/setup/minikube/) to deploy a single-node
Kubernetes cluster.
* For production, we will use a Kubernetes cluster installed by the 
[DeepOps project](https://github.com/nvidia/deepops).

## Prerequisites

* [Kubernetes v1.10.0](https://kubernetes.io) 
* [NVIDIA GPU Device Plugin](https://github.com/NVIDIA/k8s-device-plugin#preparing-your-gpu-nodes)
* [Helm](https://helm.sh)
* [prometheus-operator](https://github.com/coreos/prometheus-operator)
  ```
  helm repo add coreos https://s3-eu-west-1.amazonaws.com/coreos-charts/stable/
  ```

## Setup

The following packages will be installed on your Kubernetes cluster:
* [CoreOS's Prometheus Operator](https://github.com/coreos/prometheus-operator) for gathering and monitoring metrics
* [Istio v0.8](https://istio.io) for ingress and load-balancing

After the installation of those packages, we will deploy the following:
* Scalable [K8s Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
of the [TensorRT GRPC example](../02_TensorRT_GRPC) `inference-grpc.x` 
* YAIS specific instance of a Prometheus server that will scrape any Pods labeled `scrape: yais`
* Istio `Gateway` and `VirtualService` to route load-balanced traffic to our gRPC service.

## Install

Starting at this point, you should have a Kubernetes cluter with all the prerequisites.

If you use the [minikube setup](minikube/README.md) you can simply run:
```
./bootstrap-minikube.sh
```

Otherwise, you can choose to install each of the components manually.

### Prometheus Operator

Initialize Helm and install the `prometheus-operator` and `kube-prometheus`

```
cd ../prometheus
./bootstrap.sh
cd ..
```

Monitor `kubectl get pods -n monitoring` and wait everything to come up.

Customize any settings in the [custom-settings.yml](prometheus/custom-settings.yml)
file.  This project is exposing the Grafana server as a `NodePort` and providing custom 
datasource and dashboards for YAIS metrics.

### Istio

Initialize Istio.  I've provided the Istio v0.8 `istio-demo.yml` modified to use a `NodePort`
as `istio/minikube.yml`  If you are using a cloude instance, you can change to a `LoadBalancer`.

```
kubectl create namespace istio-system
kubectl apply -f istio/istio-v1.0-minikube.yml
kubectl label namespace default istio-injection=enabled
```

### YAIS Service

```
kubectl apply -f yais-deploy.yml
```

This does the following:
* `Deployment` - launches the service and resources
* `Service` - provides access policy to the deployment pods
* `ServiceMonitor` - tells our Prometheus server to scrape YAIS metrics
* `Gateway` - ingress host, port and protocol
* `VirtualService - routing ingress to services

### Test the Service

Use the [`devel.sh`](devel.sh) script in the project's root directory.

```
# from project root
./devel.sh
cd build/examples/02_TensorRT_GRPC
./siege.x --port 31380 --rate=1000
```

`31380` is the default `NodePort` for the Istio `ingressgateway`.

Note: If you get errors, sometime it takes a short period of time before the ingress gateway
is updated to reflect the routing. 

### Check the Metrics

```
kubectl get svc -n monitoring | grep grafana
```

The default login is `admin/admin`.  Navigate to the `YAIS` dashboard.  Celebrate.
