# Envoy Load Balancer

Very basic Envoy Proxy L7 load balancer for testing purposes.

`run_loadbalancer.py -n <number of instances>` will start a copy of envoy loadbalancing
over ports `[50051, 50051+n-1]`.  You are responsible for spinning up the backend services.

