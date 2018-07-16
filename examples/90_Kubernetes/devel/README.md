
## Round 1: External Service

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

## Round 2: Package and Deploy

TODO - We could use some community help here.

## Round 3: Optimize Deploy Contaienr

TODO - We could use some community help here.
