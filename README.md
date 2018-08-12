# YAIS

(Pronounced: Yaz!)

C++ library for developing compute intensive asynchronous services built on gRPC.

YAIS provides a bootstrap for CUDA, TensorRT and gRPC functionality so developers
can focus on the implementation of the server-side RPC without the need for a lot of
boilerplate code.

Simply implement define the gRPC service and request/response `Context` and an associated set of `Resources`.

## Quickstart

### Prerequisites
  * Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
  * Sign-up for [NVIDIA GPU Cloud](https://ngc.nvidia.com/) and acquire an [API Key](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html#generating-api-key).
  * Authenticate your Docker client using your NGC API key. Yes, the username is `$oauthtoken`.

```
$ docker login nvcr.io
Username: $oauthtoken
Password: <paste-your-ngc-api-key-here>
```

### Clone and build YAIS

```
git clone https://github.com/NVIDIA/yais.git
cd yais
make
./devel.sh
./build.sh
```

The above commands build a docker image, maps the current working directory inside the container,
and finally, builds the library inside the container.  All dependencies are provided by the container, 
but the actual source code remains on the host.  For deployment, copy or build the library as part
of the container's filesystem.

### Compile Models

Next, compile the supplied [models](models) using [TensorRT 4](https://developer.nvidia.com/tensorrt):

```
cd models
./setup.py
```

Modify the setup.py file to choose the models, batch sizes, and precision types you wish to build.
The default is configured for a Tesla V100.  Not all GPUs support all types of precision.

### Run Examples

Finally, run the `inference.x` executable on one of the compiled TensorRT engines.

```
root@dgx:/work/models# /work/build/examples/00_TensorRT/inference.x --engine=ResNet-50-b1-int8.engine --contexts=8
I0703 06:08:20.770718 13342 TensorRT.cc:561] -- Initialzing TensorRT Resource Manager --
I0703 06:08:20.770999 13342 TensorRT.cc:562] Maximum Execution Concurrency: 8
I0703 06:08:20.771011 13342 TensorRT.cc:563] Maximum Copy Concurrency: 16
I0703 06:08:22.345489 13342 TensorRT.cc:628] -- Registering Model: 0 --
I0703 06:08:22.345548 13342 TensorRT.cc:629] Input/Output Tensors require 591.9 KiB
I0703 06:08:22.345559 13342 TensorRT.cc:630] Execution Activations require 2.5 MiB
I0703 06:08:22.345568 13342 TensorRT.cc:633] Weights require 30.7 MiB
I0703 06:08:22.392627 13342 TensorRT.cc:652] -- Allocating TensorRT Resources --
I0703 06:08:22.392644 13342 TensorRT.cc:653] Creating 8 TensorRT execution tokens.
I0703 06:08:22.392652 13342 TensorRT.cc:654] Creating a Pool of 16 Host/Device Memory Stacks
I0703 06:08:22.392663 13342 TensorRT.cc:655] Each Host Stack contains 608.0 KiB
I0703 06:08:22.392673 13342 TensorRT.cc:656] Each Device Stack contains 3.2 MiB
I0703 06:08:22.392680 13342 TensorRT.cc:657] Total GPU Memory: 52.0 MiB
I0703 06:08:22.506074 13342 inference.cc:93] -- Inference: Running for ~5 seconds with batch_size 1 --
I0703 06:08:27.511525 13342 inference.cc:131] Inference Results: 11898 batches in 5.00546 seconds; sec/batch: 0.000420698; inf/sec: 2377
```

The best way to explore YAIS is to dig into the [examples](#examples).

## Motivation

The following is a quote from the [gRPC multithreading documentation](https://github.com/grpc/grpc/pull/10919/files).
The essence of which states, asynchronous threading models provide the best possible performance, but are intrinstically difficult to program.

The goal of this library is to minimize the boilerplate code need, while providing a simple, yet opinionated, approach
to building highly scalable compute bound microservices.

>### Asynchronous API Threading model
>  * The asynchronous model is considerably difficult to work with but is the right tool of choice when finer control on the threading aspect of the rpc handling is desired. In this model, gRPC does not create any threads internally and instead __relies on the application to handle the threading architecture__. 
>  * The application tells gRPC that it is interested in handling an rpc and provides it a completion key(`void*`) for the event (a client making the aforementioned rpc request).
>  * The application then calls `Next` (potentially a blocking call) on completion queue waiting for a completion key to become available. Once the key is available, the application code may react to the rpc associated with that key by executing the code that it chooses. The rpc may require more completion keys to be added to the completion queue before finishing if it is a streaming rpc, since each read or write on the stream has its own completion event.
>  * **Pros**
>    * Allows the application to bring in it’s own threading model.
>    * No intrinsic scaling limitation. Provides best performance to the integrator who is willing to go the extra mile.
>  * **Cons**
>    * The API needs considerable boilerplate/glue code to be implemented. 

## Overview

There are two fundamental classes of the YAIS library:
  * `Context`
    * Defines the logic of the RPC function
    * Maintains the state of the transaction throughout the lifecycle of the RPC. 
  * `Resources`
    * Provides any resources, e.g. threadpools, memory, etc. needed by
      a Context to handle the business logic of the RPC.

For a high-level overview see the [YAIS Slide Deck](https://docs.google.com/presentation/d/1n0g082jMJfq72dxbef9bThn4mQbzWKG6TzirJIuCEjE/edit?usp=sharing).

For details on the integrated convenience classes, see the [Internals document](examples/10_Internals/README.md). 

## Examples

### TensorRT

Code: [TensorRT](examples/00_TensorRT)

Example TensoRT pipline. This example uses fixed-size resources pools for input/output 
tensors and execution contexts. Thread pools are used to provide maximum overlap for 
async sections. Concurrency is limited by the size of the resource pools.  If a resource
pool is starved, only that part of the pipeline will stall.

### Basic GRPC Service

Code: [Basic GRPC Service](examples/01_Basic_GRPC)

Simple Ping-Pong application to demonstrate the key gRPC wrappers provided by YAIS.
The [server.cc](examples/01_GRPC/src/server.cc) provides the best documentation of how
a YAIS server is structured.

### Inference Service

Code: [TensorRT GPRC Service](examples/02_TensorRT_GRPC)

Combines the ideas of the first two examples to implement the compute side of an inference
service.  The code shows an example (`/* commented out */`) on how one might connect to an
external data service (e.g. an image decode service) via System V shared memory.

This example is based on our flowers demo, but simplified to skip the sysv ipc shared memory 
input buffers.  Typically this demo is run using ResNet-152, in which case, the compute is
sufficiently large not to warrant a discrete thread for performing the async H2D copy. Instead
the entire inference pipeline is enqueued by workers from the CudaThreadPool.

### Internals

Code: [Internals](examples/10_Internals)

The `internals.cc` and [README](examples/10_Internals/README.md) provide a guide on the provided 
convenience classes. practice.  The sample codes builds a NUMA aware set of buffers and threads.
For implementation details, go directly to the [source code](yais).

### Kubernetes

The [Kubernetes example](examples/90_Kubernetes) might be useful for those who are new to developing
a Kubernetes application.  We start by developing and building in a docker container. Next, we 
link our development container as an external service to a minikube cluster to expose metrics.

The remaining items are TODOs:
  - Build an optimized deployment container
  - Demostrate Envoy load-balancing
  - Use Istio to simplify the load-balacing deployments
  - Use metrics scraped by Prometheus to trigger auto-scaling

### Execution Models

There are two primary ways on which you can expose concurrency in deployment.  The preferred
method is to use a [single process with multiple streams](examples/97_SingleProcessMultiStream).
Alternatively, one can use [NVIDIA's Multi-Process Server, aka MPS](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf).

Single Process / Multiple Streams is preferred for:
  * serving models that are logically grouped
  * simplifed and more efficient memory management

#### Single Process / Multiple Streams (SPMS)

#### Multiple Process Server (MPS)

Code: [MPS Example](examples/98_MultiProcessSingleStream)

Tests multiple TensorRT inference services running on the same GPU. `N` services are started, a
load-balancer is created to round robin incoming requests between services, and finally a client
sending 1000 requests to the load-balancer and measures the time.  The `run_throughput_test`
starts a subshell after the initial 1000 requests have been sent.

#### Clients

Three clients are available:
  * `client-sync.x` - send a blocking inference request to the service and waits for the
     response.  Only 1 request is ever in-flight at a given time.
  * `client-async.x` - the async client is capable of issuing multiple in-flight requests.
     Note: the load-balancer is limited to 1000 outstanding requests per client before circuit-
     breaking.  Running more than 1000 requests will trigger 503 if targeting the envoy load-
     balancer.
  * `siege.x` - constant rate (`--rate`) async engine that is hard-coded to have no more than
     950 outstanding in-flight requests.  A warning will be given client-side if the outstanding
     requests tops meaning the rate is limited by the server-side compute.

This examples makes use of an [Envoy proxy](https://github.com/envoyproxy/envoy) which is configured
by a template in the [Load Balancer example](examples/99_LoadBalancer).

### DALI feeding TensorRT

WIP

### NVVL

WIP

## Copyright and License

This project is released under the [BSD 3-clause license](LICENSE).

## Issues and Contributing

* Please let us know by [filing a new issue](https://github.com/NVIDIA/YAIS/issues/new)
* You can contribute by opening a [pull request](https://help.github.com/articles/using-pull-requests/)

Pull requests with changes of 10 lines or more will require a [Contributor License Agreement](CLA).

> YAIS: Yet Another Inference Service Library
