# TensorRT GRPC Example

This examples extends the [TensorRT](examples/00_TensorRT) compute loop into an
async gRPC service similar to [example 01_gRPC](examples/01_GRPC).

There are three take-aways from this example:

1. TensorRT compute pipeline is implemented as the `ExecuteRPC` virtual function
   of the `Context`.
2. An external datasource is used to override the input bindings
3. Custom [Prometheus](https://prometheus.io) metrics for inference compute and
     request durations, load ratio, and GPU power gauge.
     are recorded/observed.

## Quickstart

```
cd /work/build/examples/02_TensorRT_GRPC
./inference-grpc.x --contexts=8 --engine=/work/models/ResNet-50-b1-int8.engine --port 50051 &
./siege.x --port=50051 --rate=2500
# ctrl+c to cancel client
telegraf -test -config /work/examples/91_Prometheus/scrape.conf
```

## Explore

Fun things to try:

  * Evaluate the performance of the model using `inference.x` in 
    [examples/00_TensorRT](examples/00_TensorRT)
  * Try running `siege.x` below, at, and above the benchmarked rate and watch the metrics
    via `telegraf`.
  * Deploy on Kubernetes, collect metrics via Prometheus and visualize using Grafana;
    [examples/90_Kubernetes](examples/90_Kubernetes).

## Server/Service

`inference-grpc.x` CLI options:

  * `--engine` - the compiled TensorRT plan/engine
  * `--contexts` - the maximium number of concurrent evaluations of the engine.
  * `--port` - the port on which requests are received (default: 50051)
  * `--metrics` - the port on which to expose metrics to be scraped (default: 50078)


## Clients

Three clients are available:
  * `client-sync.x` - sends a blocking inference request to the service and waits for the
     response.  Only 1 request is ever in-flight at a given time.
  * `client-async.x` - the async client is capable of issuing multiple in-flight requests.
     Note: the load-balancer is limited to 1000 outstanding requests per client before circuit-
     breaking.  Running more than 1000 requests will trigger 503 if targeting the envoy load-
     balancer.  The client has no backoff and will try to send the full compliment of requested
     inference requests.  `siege.x` is the better async client.
  * `siege.x` - constant rate (`--rate`) async engine that is hard-coded to have no more than
     950 outstanding in-flight requests.  A warning will be given client-side if the outstanding
     requests tops meaning the rate is limited by the server-side compute.

TODO:
  * Add more varied test clients akin to [Netflix's Chaos Monkeys](https://github.com/Netflix/chaosmonkey),
    but for gRPC client behavior.
    * Random rate, random pulses, canceled messsages, messages wiht unreasonable timeouts, etc.

## Metrics

YAIS metrics are gathered and exposed via the [prometheus-cpp](https://github.com/jupp0r/prometheus-cpp) 
client library.  In this example, we expose four custom 
[metrics](https://prometheus.io/docs/concepts/metric_types/): 2 Summaries, 1 Histogram and 1 Gauge.

  * `compute_duration` and `request_duration` are summaries recored with the model
     name as a component of the metric.  This is useful for evaluating how a given
     model is performing, but this is not a good metric to aggregate across multiple
     service.
  * `load_ratio` is a histogram of `request_duraton / compute_duration`.  Ideally, this
     unitless value is just over 1.0.  Values higher than 1.0 are indictive of some
     delays in the compute of a given request. Sources of delays include, overloaded
     queues and/or starvation of resources. Histograms can be aggregated across services,
     which makes this metric a good candidate for triggering auto-scaling.
  * `gpu_power` is a simple gauge that periodicly reports the instaneous power being
    consumed by the device.  As the load increases on the service, the power should 
    increase proprotionally, until the power is capped either by device limits or compute 
    resources. When power capped, the `load_ratio` will begin to increase under futher 
    increases in traffic.


### Acquiring Metrics

Prometheus metrics are generally scraped by a Prometheus service.  When using Kubernetes
to deploy services, the [prometheus-operator](https://github.com/coreos/prometheus-operator)
provides a [`ServiceMonitor`](https://github.com/coreos/prometheus-operator#customresourcedefinitions)
which allows you to define custom scraping configuration per service. See the 
[Kubernetes example](examples/90_Kubernetes) for more details.

While testing, you can use the [`telegraf`](https://github.com/influxdata/telegraf) application
to scrape local services.

```
# start service
telegraf -test -config /work/examples/91_Prometheus/scrape.conf
```

Here is some sample output (line breaks added for readability):
```
> yais_inference_compute_duration_ms,host=dgx,model=flowers,url=http://localhost:50078/metrics count=1000,sum=2554.070996 1530985302000000000

> yais_inference_compute_duration_ms_quantile,host=dgx,model=flowers,quantile=0.500000,url=http://localhost:50078/metrics value=2.526903 1530985302000000000
> yais_inference_compute_duration_ms_quantile,host=dgx,model=flowers,quantile=0.900000,url=http://localhost:50078/metrics value=2.625447 1530985302000000000
> yais_inference_compute_duration_ms_quantile,host=dgx,model=flowers,quantile=0.990000,url=http://localhost:50078/metrics value=2.855728 1530985302000000000

> yais_inference_request_duration_ms,host=dgx,model=flowers,url=http://localhost:50078/metrics count=1000,sum=243547.558097 1530985302000000000
> yais_inference_request_duration_ms_quantile,host=dgx,model=flowers,quantile=0.500000,url=http://localhost:50078/metrics value=253.216653 1530985302000000000
> yais_inference_request_duration_ms_quantile,host=dgx,model=flowers,quantile=0.900000,url=http://localhost:50078/metrics value=256.715759 1530985302000000000
> yais_inference_request_duration_ms_quantile,host=dgx,model=flowers,quantile=0.990000,url=http://localhost:50078/metrics value=275.407232 1530985302000000000

> yais_inference_load_ratio,host=dgx,url=http://localhost:50078/metrics +Inf=1000,1.25=1,1.5=1,10=9,100=253,2=1,count=1000,sum=95879.013208 1530985302000000000

> yais_gpus_power_usage,gpu=0,host=dgx,url=http://localhost:50078/metrics gauge=52.821 1530985302000000000

> yais_executor_queue_depth,host=dgx,url=http://localhost:50078/metrics gauge=0 1530985302000000000
```
### Best Practices

For a good description of using histograms vs. summaries to collect meaningful metrics
see: https://prometheus.io/docs/practices/histograms/

Two rules of thumb:
 - If you need to aggregate, choose histograms.
 - Otherwise, choose a histogram if you have an idea of the range and distribution of 
   values that will be observed. Choose a summary if you need an accurate quantile, no
   matter what the range and distribution of the values is.

