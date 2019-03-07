# Routing Requests

If we multiple instances of TRTIS each with different models, we need a way to
route requets to the proper service. There are two convenient options: routing
by subdomain or routing by headers.

In this example, we will have three unique pools of TRTIS services:
- Pool A: only handles `model_a` `Classify` requests
- Pool B: only handles `model_b` requests (`Classify` or `Detection`)
- General Pool: handles all other requests

In our hypothetical deployment scenario, both `model_a` and `model_b` are
particularly active so we have dedicated resources to handle those requests.
Similarly, our general pool has it's own fixed size. Later, we will show out to
auto-scale pods based on TRTIS and GPU metrics.

We want our service to be hosted on a single endpoint `trt.lab`.

To test routing, we have provided an `envoy.yaml` configuration which using the
local copy of envoy in the container. We will start a load-balancer/router on port
50050 and route to three sample services running on 51051, 51052 and 51053.

In order to route by model without deserializing the gRPC requests, we've asked
gPRC to add custom-metadata (http headers) to the request. See the ImageClient
for more details.

```c++
std::map<std::string, std::string> headers = {{"custom-metadata-model-name", model_name}};
```

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

While we are using Envoy (v1.9) directly in this example, we will later show how
this can be accomplished in Istio. A major TODO in this project is to build a
TRTIS operator while will provide Kubernetes CRD that will be able to
dynamically manage the routes as a function of where the model will be loaded on
the cluster. Whereas this example shows static placement, we eventually want to
get to fully dynamic routes.