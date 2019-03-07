# Deploying Inference Services

This document/example folder is a work in progress. Its intent is to cover
various aspect of deployment including strategies, limitations, services and
kubernetes examples of deploying inference services.

Over the course of this guide, we will build a full end-to-end image processing
service deployed on Kubernetes.

Let us start by assuming all your models can be served/deployed with the
[TensorRT Inference Server, aka TRTIS](https://github.com/nvidia/tensorrt-inference-server).
One of the primary advantages of TRTIS is the ability to host multiple models in
a single linux process. Given the capabilities of modern GPUs, this is the most
efficient way to for multiple models to efficiently share of both compute and
memory resources.

Next, lets dive a little deeper into the features of TRTIS that gives this effiency advantage.

  - _Concurrent Executions_ allow for multiple independent inference batches to be
    in-flight on the device at a given time. This could be multiple batches of
    the same model or single batches of different models, or any combination
    imaginable. 
    - Running on a Tesla V100 GPU with ResNet-152 Batch8
      - Allowing only 1 in-flight batch8 yields XXX images/sec with a compute
        latency of YYY.
      - Allowing 8 concurrency batch8 executions increases the throughput to
        2500 images/sec; however, the compute latency per batch increases to
        ZZZ.
    - To evaluate the performance of TensorRT models a function of concurrent
      in-flight executions we provide the [TensorRT/ConcurrentExecution](../TensorRT/ConcurrentExecution)
      example.
      ```bash
      infer.x --engine=/external/models/ResNet-152-b8-fp16.engine --concurrency=8
      ```
    - The value of concurrent executions will differ depending on the compute
      requirements of the model and the GPU on which it executes. The best
      practice is to benchmark and evaluates performance.

  - _Tunable Concurency_ enables you to specify on a per model basis the number
    of concurreny copies that can be executed at any given time. Follow the guidelines
    Concurrent Executions to tune this option on a per model basis.

  - _Dynamic Batching_ allows individual requests from either same or different
    clients to be multiplexed into a single mini-batch and infered. Dynamic
    Batching is performed on a per-model basis and can have a max
    `preferred_batch_size` and a `max_queue_delay_microseconds` which specific
    how long of a delay messages are allowed to accumulate. Batching is one of
    the best ways to improve throughput. Depending on your needs, you will want
    to balance the added latency required for batching vs the throughput
    improvements achieved.
    - Note: in scale-out deployments where unary (send/recv, not streaming)
      requests are being load-balanced across multiple TRTIS instances, the
      value of dynamic batching in the TRTIS decreases as the number of replicas  
      increases.
    - TODO: We address this issue by creating Dynamic Batching Services that sit
      infront of the Load-Balancer. Add a discussion and update the [examples/03_Batching]
      example with the latest streaming Server/Client

  - _Custom Metrics_ provide application-level metrics on how the TRTIS service
    is performing. Analyzing these metrics can provide insight on when a service
    is being overloaded and when to add more resources. TRTIS and some TRTLAB
    examples expose Prometheus Metrics.
     - [examples/90_Kubernetes/prometheus] is one exmaple of how to use
       Kubernetes + Prometheus + Grafana to scrape and visualize metrics from
       running TRTIS services.
     - TODO: Document and clean up examples.

The goal of this project is to provide supplementary support to TRTIS by
providing buildings blocks that help you build companion microservices that
worth with TRTIS, as well as example deployment scenarios.

## Configure a TRTIS Kubernetes Deployment

In Kubernetes, a
[Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
allows you to define a [Pod](https://kubernetes.io/docs/concepts/workloads/pods/pod/)
and the number of copies of that pod, i.e.
[ReplicaSet](https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/)
that you would like deployed on your cluster.

### Build a Model Store

The [TRTIS Model Store] is the collection of models that will be loaded and
served by a TRTIS instance. For details on how to build a model store, see the [Deployment/ModelStore]
example.

In order to define our Kubernetes Deployment, we need to map our model store
into our TRTIS Pod. There are multiple ways this can be achieved.
- Extend the TRTIS container image and add the model store to the image
- Mount an External Volume into the TRTIS Pod
- Use another container in the same TRTIS Pod that will dynamically generate a
  model store from some external data sources, e.g. S3.
  - TODO: Add this example

My Kubernetes cluster has an NFS mount at `/shared` on every node. For this
example, our model store will be located at `/shared/trtis/example-model-store`

### Deploy TRTIS Pods




## Configure Multiple TRTIS Deployments with Different Sets of Models

Now suppose you are serving more models than you can allocate on a single GPU.  In this scenario, we can split our list of models into groups and spread those groups out over multiple TRTIS `Deployments`

## Scenario #1: 2 Models, 10 Servers, 20 GPUs

Assume you have two computer vision models, e.g. classification, object
detection, segmentation, that you wish to deploy on 10 servers.

Probably the first questions you might ask yourself is what is the breakdown of
load expected for each model. Is it 50/50? Or is it 90/10? Does it vary by time
of day? Is it predictive?

The strategy with TRTIS is simple, you deploy 20 replicas of the TRTIS service
across 10 servers, 1 TRTIS service per GPU and tell your load-balancer to
round-robin requests across your services.

However, unless you customize TRTIS, by default, TRTIS only accepts raw tensors
as inputs and returns raw tensors an outputs.

What are your inputs?  And what's a reasonable expectation for the rest of the
inference pipeline?

Let's assume you are receiving JPEG images. First, you need to decode the images to
raw pixels, then you need to prepare the images to be inferred. To keep things
simple, let's assume that both models use the same input preprocessing method.

What is the compression ratio of your JPEG images? Assuming the images are
8-bit, then
[a blog post by Graphics Mill](https://www.graphicsmill.com/blog/2014/11/06/Compression-ratio-for-different-JPEG-quality-values#.XHtdPpNKiXE_)
measured the compression ratio to be 1:5.27 for JPEG Quality 100 (Q=100) and
1:43.27 for Q=55. For this discussion we wil focus on Q=80, which was measured
to be a 1:25 ratio. This is the decompression ratio for JPEG bytes to INT8 pixel
values. However, most DNN models after normalization accept tensors that are
fp32 values as inputs. This means that we have a 1:100 an increased ratio of
input bytes to bytes of input tensors.

A 100KiB image becomes a 9.75MiB data structure that needs to be provided to
TRTIS. This 100x increase in size is big, but not unreasonably large. If your
images are coming in over the Internet/WAN, then your LAN connction is likely to
be 100x faster. However, if this was video, that compression ratio per frame
would be MUCH larger. To future-proof our implementation, we are going to do our
best to minimize the amount of data we move around.

Next, what is it that our users are providing to us beside just the image to be
inferred. The client must inform the server which model should be used for the
request. And similarly, the server probably also wants to know some details
about the user. In this example, we will using a user API key to authenticate
the user.

Now let's look at the data payload the client will be sending our server. We are
going to break down an image inference request into two parts:
- the bulk image data
- the request metadata: api key, image_uuid, model_name

This separation allows us to commmunicate and move each component more
optimally. We will send the JPEG image bytes directly to an object store and we
will use a gRPC message to communicate the metadata for the request.

Why not embed the image directly into the gRPC message? While this is certainly
a possibility, we are chosing this separation because in our optimial pipeline,
our client request message will go through several services before the bulk
image data is needed. These services include an ingress router/load-balancer and
an external batching service before they are sent to the image pre-processing
service. This separate helps us future-proof our implementation by avoiding
unnecessary data movement.

Let's break down our client implementation:
- 1) writes the jpeg image data to an S3-compatible object store as some UUID.jpg
- 2) creates an async gRPC unary (send/recv) RPC request to our inference service.
  - the message payload consists of our client api key, image uuid and model_name
  - add custom headers that will enable our ingress router/load-balancer to
    properly route the message to the correct target without the need to
    deserialize the request payload. this allows us to route message directly to
    services specify to some metadata. In this case, we will add the model name
    to the headers so we can eventually route requests to batching services
    unique to that model.

This means our client can continue to issue async inference requests with the
promise that the results will returned in some future time.

On the server side, this data move break down the inference request into two
components, the data and the request.

Let's assume our incoming data transport is very effficient. The images will
be deposited into an S3-compatible object store.  On my Kubernetes cluster,
I'll be using [Rook's Mino/S3 Operator](https://rook.io), but it would work
equally well on AWS.

