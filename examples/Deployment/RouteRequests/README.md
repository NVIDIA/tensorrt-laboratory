# Routing Requests

If we have multiple instances of TRTIS each with different models, we need a way
to route requests to the proper service. There are two convenient options:
routing by subdomain or routing by headers.

In this example, we will have three unique pools of TRTIS services:
- Pool A: only handles `model_a` `Classify` requests
- Pool B: only handles `model_b` requests (`Classify` or `Detection`)
- General Pool: handles all other requests

In our hypothetical deployment scenario, both `model_a` and `model_b` are
particularly active so we have dedicated resources to handle those requests.
Similarly, our general pool has it's own fixed size. Later, we will show out to
auto-scale pods based on TRTIS and GPU metrics.

A simple approach would be to host `model_a.trt.lab`, `model_b.trt.lab` and
`general_pool.trt.lab` and have the client make the decision sender side on
where to send the requests. However, changest to the service layout would
require updates to the clients software which makes this option less appealing.

Ideally, we want our entire service to be hosted on a single endpoint `trt.lab`.  

To ensure our requests arrive at the proper destination server-side, we have our
client add [Custom Metadata](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md) to each gRPC request.

```c++
std::map<std::string, std::string> headers = {{"custom-metadata-model-name", model_name}};
```

which inside our client library

```c++
for (auto& header : headers)
{
    // add headers to ::grpc::ClientContext
    ctx->m_Context.AddMetadata(header.first, header.second);
}
```

To test routing, we have provided an `envoy_config.yaml` configuration. The
load-balancer/router listens on port 50050 and routes to three sample services
running on 51051, 51052 and 51053.

To differentiate by endpoint, `Classify` or `Detection`, we can match routes on their `uri`.

Here is the relevant parts of the envoy config:
```yaml
- match:
    prefix: /trtlab.deploy.image_client.Inference/Classify
    headers:
    - name: custom-metadata-model-name
      exact_match: model_a
    grpc:
  route:
    cluster: classify_model_a
- match:
    prefix: /
    headers:
    - name: custom-metadata-model-name
      exact_match: model_b
    grpc:
  route:
    cluster: model_b
- match:
    prefix: /
    grpc:
  route:
    cluster: general_pool
```

`test_routing.sh` provides a convenient means to test the configuration. It will
compile a simple implementation of the ImageClient service, bring up an instance
of Envoy, 3 instances of the test_service, then send both `Classify` and
`Detection` requests with three different models at the router. The service
implementation simple returns which named service handled the request.

```s
root@5e8ffb38df87:/work/examples/Deployment/RouteRequests# ./test_routing.sh 
... some start up output ...
Testing Classify RPC
I0307 13:41:36.614954   355 test_service.cc:74] model_a served by model_a
I0307 13:41:36.616070   359 test_service.cc:74] model_b served by model_b
I0307 13:41:36.617031   362 test_service.cc:74] model_c served by general_pool
Testing Detection RPC
I0307 13:41:36.617636   362 test_service.cc:74] model_a served by general_pool
I0307 13:41:36.618005   359 test_service.cc:74] model_b served by model_b
I0307 13:41:36.618367   362 test_service.cc:74] model_c served by general_pool

**** Test Passed ****
```

While we are using Envoy (v1.9) directly in this example, we will later show how
this can be accomplished in Istio. A major TODO in this project is to build a
TRTIS operator while will provide Kubernetes CRD that will be able to
dynamically manage the routes as a function of where the model will be loaded on
the cluster. Whereas this example shows static placement, we eventually want to
get to fully dynamic routes.