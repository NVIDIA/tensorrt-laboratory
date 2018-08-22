# Batching Service

A batching service is a service that trying to collect sets of similar requests into a
collective batch which can be executed in a single-shot.

#### Why do we want to batch?

In the case of Deep Neural Networks, batching can improve the computational efficiency
of executing on a GPU by increases the operational intensity, i.e. improving the ratio of 
the number of math operations per memory transaction.  This translates to improved
throughput, better hardware utilization and cost reductions.

#### Sounds great, but what's the catch?

In many cases, batching can add latency to an individual request.  Because a batch of more
than 1 item, BatchN, computed as a single unit, the time to compute BatchN is greater than
Batch1.  However, in many cases, the time to compute delta between Batch1 and Batch2/4/8 is 
fairly small due to the improved operational efficiency.

Secondly, because batching requires requests to be collected, there is a timed collection
window prior to the compute.  The first request in a batch sees the longest latency. 

The worst-case increased latency is bounded by the following formula:
```
worst_additional_latency = batch_window_timeout + batchN_compute - batch1_compute
```

#### When to Batch?

You want to batch requests when your service has very high-load and you can tolerate
minor increases in latency.

Throughput improvements can be 2-5x which translates into direct cost savings.

#### What does this Batching Service do for me?

The basis YAIS service examples [01_GRPC](../01_GRPC) and [02_GRPC_TensorRT](../02_GRPC_TensorRT)
have implemented high-performance send/recv unary services.  That is, the client
sends a request which is computed and a response is returned.  The client could in theory
create a single message is itself a batch, i.e. multiple images files or sentences to be 
translated.  However, in most common realworld usecases, the clients of a service send
a single item at a time.  This keeps the logic simple and lifecycle of the request simple.

If this is your RPC definition,
```
service Inference {
   rpc Compute (Input) returns (Output) {}
}
```

Then, instead of implementing `rpc Compute` to perform the inference computation, instead, we
hyjack that RPC and turn it into a batcher.  In the [`inference-batcher.cc`](inference-batcher.cc)
file, you will see that we are indeed we implement our batching service as the `Compute` method.

The batching service collecting incoming `Input` requests and forwards them via a gRPC stream
to a service that accepts a "batching stream".

A “batching stream” is a stream where the endpoint service reads and collects the elements of the stream until the client signifies it is done writing.  That is the signal at which YAIS performs a single batched inference call on the concatenated set of requests that came in over the stream.  After the inference calculation is complete, the server writes the results for each request item to the stream.  That is, for each request that came in on the stream, the server is expected to return a response.  

We still need to compute inference on the batching stream.  This is performed by [streaming-service.cc](streaming-service.cc).

The `streaming-service` implements the `BatchedCompute` RPC method using a `BatchingContext`.
```
service Inference {
   rpc Compute (Input) returns (Output) {}
   rpc BatchedCompute (stream Input) returns (stream Output) {}
}
```

Because the stream consists of an array of individual messages, you simply need to make
minor modifications to your existing Batch1 service to preprocess and concat the incoming requests
together to form a single batch compute.  For each `Input` item in the stream, it is expected that
the service writes an `Output` response in the same order as the inputs (FIFO).

The batching service doesn’t need to know anything about the format of the `Input`/`Output` messages.  It simply accepts and forwards them.  The result is that this batching service example should be able to work with any unary gRPC service with any request/response message.  You simply need to implement a streaming service capable of handling the forwarding stream.

## Running Example

```
./launch_batching.sh
```

```
... # streaming service startup
... # batching service startup

Starting a shell keeping the services and load-balancer running...
Try python unary_client.py - exit shell to kill services
Batching Subshell: python unary_client.py
I0822 14:48:18.900671    50 inference-batcher.cc:344] incoming unary request
I0822 14:48:18.902642    41 inference-batcher.cc:109] Client using CQ: 0x14470f0
I0822 14:48:18.902680    41 inference-batcher.cc:140] Starting Batch Forwarding of Size 1 for Tag 0x1458450
I0822 14:48:18.903472    35 streaming-service.cc:61] Recieved request with batch_id=78
I0822 14:48:18.903504    35 streaming-service.cc:54] Response with batch_id=78
I0822 14:48:18.903656    47 inference-batcher.cc:243] Batch Forwarding Completed for Tag 0x1458450
Received msg  with batch_id=78
```
