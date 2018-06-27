# YAIS

C++ library for compute intensive asynchronous services built on gRPC.

> Yet Another Inference Service Library

## Quickstart

```
git clone ssh://git@yagr.nvidia.com:2200/demos/inference-demo/grpc_inference_service_cpp.git
cd grpc_inference_service_cpp
make
./devel.sh
./build.sh
cd build
```

[See the benchmarking example for more details](src/Examples/01_Benchmark/README.md).


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

For details on the integrated convenience classes, see the [Internals document](docs/Internals.md). 

## Examples

### Echo Service

Code: [Echo Service](src/Examples/00_Echo)

Ping-pong / Echo message service based on gRPC.  Describes the core messaging components of the
library.

### TensorRT Perf Test

Code: [TensorRT Perf Test](src/Examples/01_Benchmark)

Example TensoRT pipline with fixed resources pools for input/output tensors and execution
contexts.  Thread pools are used to provide maximum overlap for async sections such that 
if a resource pool is starved, only that part of the pipeline will stall.

### Inference Service

Code: [Inference Service](src/Examples/02_Flowers)

Formerly the flowers demo, simplified to skip the sysv ipc shared memory input buffers.  Not as much
concurrency as the perf test/ benchmark code above.

### MPS

Code: [MPS Example](src/Examples/98_MPS)

Tests multiple TensorRT inference services running on the same GPU. `N` services are created, a
load-balancer is created to round robin incoming requests between services, and finally a client
sending 1000 requests to the load-balancer and measures the time.

### DALI feeding TensorRT

TODO: Build this!

### NVVL

TODO: Build this!

## Copyright and License

This project is released under the [BSD 3-clause license](LICENSE).

## Issues and Contributing

A signed copy of the [Contributor License Agreement](CLA) needs to be provided to <a href="mailto:rolson@nvidia.com">Ryan Olson</a> before any change can be accepted.

* Please let us know by [filing a new issue](https://github.com/NVIDIA/YAIS/issues/new)
* You can contribute by opening a [pull request](https://help.github.com/articles/using-pull-requests/)
