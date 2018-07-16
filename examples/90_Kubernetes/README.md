# Kubernetes

<<<<<<< HEAD
Using [Kubernetes on NVIDIA GPUs, aka KONG](http://developer.nvidia.com) is a great
way of deploying GPU accelerated microservices.  This page will act as a guide for
for both development and production deployment.

* For development, we will use [minikube](minikube.io) to deploy a single-node
=======
Using [Kubernetes on NVIDIA GPUs, aka KONG](https://developer.nvidia.com/kubernetes-gpu) is a great
way of deploying GPU accelerated microservices.  This page will act as a guide for
for both development and production deployment.

* For development, we will use [minikube](https://kubernetes.io/docs/setup/minikube/) to deploy a single-node
>>>>>>> devel
Kubernetes cluster.
* For production, we will use a Kubernetes cluster installed by the 
[DeepOps project](https://github.com/nvidia/deepops).

<<<<<<< HEAD
## Configuration
=======
## Prerequisites
>>>>>>> devel

* [Kubernetes v1.10.0](https://kubernetes.io) 
* [NVIDIA GPU Device Plugin](https://github.com/NVIDIA/k8s-device-plugin#preparing-your-gpu-nodes)
* [Helm](https://helm.sh)
<<<<<<< HEAD
* [coreos/prometheus-operator](https://github.com/coreos/prometheus-operator)

### Development with Minikube

#### Install `minikube`, `kubectl`, and `helm`

This only need to be done one time, or periodically if you wish to upgrade.

```
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/

curl -Lo kubectl https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl && chmod +x kubectl && sudo mv kubectl /usr/local/bin

curl https://raw.githubusercontent.com/kubernetes/helm/master/scripts/get > get_helm.sh
chmod 700 get_helm.sh
./get_helm.sh
```

#### Launch a Kubernetes Cluster

```
./bootstrap-linux.sh
```

Check configurations:
```
kubectl get all
kubectl get all --all-namespaces
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPUs:.status.capacity.'nvidia\.com/gpu'
# last command should report the number of GPUs on your system
# this make take some time - coffee?!
```

Initialize Helm and install the `prometheus-operator` and `kube-prometheus`
```
helm init
# wait for tiller to come up
helm install coreos/prometheus-operator --name prometheus-operator --namespace monitoring
helm install  --name kube-prometheus  coreos/kube-prometheus -f prometheus-settings.yml --namespace monitoring
```

Again, monitor `kubectl get pods -n monitoring` and wait everything to come up

#### Launch YAIS monitoring

```
kubectl apply -f yais-metrics.yml
```

At this point, you'll have the following running in the `default` namespace

```
ryan@dgx:/shared/inference/yais/examples/90_Kubernetes$ kubectl get all
NAME                            READY     STATUS    RESTARTS   AGE
pod/prometheus-yais-metrics-0   3/3       Running   1          1h

NAME                          TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)     AGE
service/kubernetes            ClusterIP   10.96.0.1       <none>        443/TCP     1h
service/prometheus-operated   ClusterIP   None            <none>        9090/TCP    1h
service/yais-metric          ClusterIP   10.96.124.141   <none>        9090/TCP    1h

NAME                                       DESIRED   CURRENT   AGE
statefulset.apps/prometheus-yais-metrics   1         1         1h
```

#### Configure Grafana

TODO:
* Add a Grafana Datasource as part of the Helm configuration
  * Unfortunately, right now I can't find the Dataset by DNS.  The WAR is to use the ClusterIP of
    the `yais-metrics` service, but this is not known apriori.  
* Refine the Grafana dashboard and include it as part of the Helm install.

At this point, you are all setup, but have no metrics to scrape.  Next we will spin up some 
YAIS services and generate some metrics.

#### Round 1: External Service

Before deploying a YAIS service with Kubernetes, we will first setup a developer environment
where we will execute our service in a Docker development container. We can still use our
Kubernetes/Prometheus/Grafana environment to gather and visualize metrics.  To do so, we
will create an external service pointing at our host.

Edit the `yais-devel.yml` and modify the IP address of the `Endpoint` to point at the host machine running
minikube (`sudo minikube ip`). 

```
apiVersion: v1
kind: Endpoints
metadata:
  name: yais-devel
subsets:
- addresses:
  - ip: 10.0.0.10  # <== ChangeMe
  ports:
  - name: metrics
    port: 50078
```

```
kubectl apply -f yais-devel.yml
```

This will create a Prometheus `ServiceMonitor` that will scrape from the external service, i.e. the 
Docker development container.  This is a good first start at integrating your service into Kubernetes
without having to do a full blown deployment.

Congrats, your minikube cluster now looking for services labeled `scrape: yais` and if found, will
automatically start scraped the port labels `metrics`.  

The final step is to bring an inference service online and to generate some load on that service.
Launch the YAIS developement container using the `devel.sh` script in the project's root directory.
Make sure all the examples have been built and models have been build, 
see [README::Quickstart](README.md#quickstart). 

```
cd examples/97_SingleProcessMultiSteam

root@dgx:/work/examples/97_SingleProcessMultiSteam# ./launch_service.sh 1 1 /work/models/ResNet-50-b1-fp32.engine
I0709 10:13:41.175212   468 Server.cc:37] gRPC listening on: 0.0.0.0:50051
I0709 10:13:41.175477   468 server.cc:229] Register Service (flowers::Inference) with Server
I0709 10:13:41.175492   468 server.cc:238] Register RPC (flowers::Inference::Compute) with Service (flowers::Inference)
I0709 10:13:41.175500   468 server.cc:243] Initializing Resources for RPC (flowers::Inference::Compute)
I0709 10:13:41.273568   468 TensorRT.cc:561] -- Initialzing TensorRT Resource Manager --
I0709 10:13:41.273602   468 TensorRT.cc:562] Maximum Execution Concurrency: 1
I0709 10:13:41.273609   468 TensorRT.cc:563] Maximum Copy Concurrency: 3
I0709 10:13:42.596443   468 TensorRT.cc:628] -- Registering Model: flowers --
I0709 10:13:42.596500   468 TensorRT.cc:629] Input/Output Tensors require 591.9 KiB
I0709 10:13:42.596511   468 TensorRT.cc:630] Execution Activations require 7.8 MiB
I0709 10:13:42.604210   468 TensorRT.cc:652] -- Allocating TensorRT Resources --
I0709 10:13:42.604228   468 TensorRT.cc:653] Creating 1 TensorRT execution tokens.
I0709 10:13:42.604236   468 TensorRT.cc:654] Creating a Pool of 3 Host/Device Memory Stacks
I0709 10:13:42.604248   468 TensorRT.cc:655] Each Host Stack contains 608.0 KiB
I0709 10:13:42.604256   468 TensorRT.cc:656] Each Device Stack contains 8.5 MiB
I0709 10:13:42.604264   468 TensorRT.cc:657] Total GPU Memory: 25.5 MiB
I0709 10:13:42.606546   468 server.cc:255] Initializing Executor
I0709 10:13:42.606832   468 server.cc:259] Registering Execution Contexts for RPC (flowers::Inference::Compute) with Executor
I0709 10:13:42.606889   468 server.cc:262] Running Server

warmup with client-async.x
1000 requests in 2.60522seconds; inf/sec: 383.845

Starting a shell keeping the services and load-balancer running...
Try /work/build/examples/02_TensorRT_GRPC/siege.x --rate=2000 --port=50051

1 x /work/models/ResNet-50-b1-fp32.engine Subshell:
```

Use `telegraf` and watch the scrape count; the `yais-devel` scraper is set to pull metrics every 2 seconds.
It can up to a minute or so until you see scraping from your k8s cluster.

```
1 x /work/models/ResNet-50-b1-fp32.engine Subshell: telegraf -test -config /work/examples/91_Prometheus/scrape.conf
...
> exposer_bytes_transferred,host=dgx,url=http://localhost:50078/metrics counter=0 1531131559000000000 # <== watch the counter
...
```

#### Round 2: Package and Deploy

TODO - We could use some community help here.

#### Round 3: Optimize Deploy Contaienr

TODO - We could use some community help here.
=======
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
of the [TensorRT GRPC example](examples/02_TensorRT_GRPC) `inference-grpc.x` 
* YAIS specific instance of a Prometheus server that will scrape any Pods labeled `scrape: yais`
* Istio `Gateway` and `VirtualService` to route load-balanced traffic to our gRPC service.

## Install

Starting at this point, you should have a Kubernetes cluter with all the prerequisites.

If you use the [minikube setup](examples/90_Kubernetes/minikube/README.md) you can simply run:
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

Customize any settings in the [custom-settings.yml](examples/90_Kubernetes/prometheus/custom-settings.yml)
file.  This project is exposing the Grafana server as a `NodePort` and providing custom 
datasource and dashboards for YAIS metrics.

### Istio

Initialize Istio.  I've provided the Istio v0.8 `istio-demo.yml` modified to use a `NodePort`
as `istio/minikube.yml`  If you are using a cloude instance, you can change to a `LoadBalancer`.

```
kubectl apply -f istio/minikube.yml
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
>>>>>>> devel
