# Envoy Load Balancer

Very basic Envoy Proxy L7 load balancer for testing purposes.

`run_loadbalancer.py -n <number of instances>` will start a copy of envoy 
listening on port `50050` and load-balancing over ports `[50051, 50051+n-1]`.

You are responsible for spinning up the backend services.

## Notes

The load-balancer overhead appears to be about 150us.  Running the `client-sync.x`
directly to a backend vs. through the load-balancer shows about 150us overhead per
transaction.

```
# direct
Throughput Subshell: /work/build/examples/02_TensorRT_GRPC/client-sync.x --port 50051
1000 requests in 2.69029 seconds; inf/sec: 371.707

# proxied via envoy load-balancer
Throughput Subshell: /work/build/examples/02_TensorRT_GRPC/client-sync.x --port 50050
1000 requests in 2.8411 seconds; inf/sec: 351.977
```