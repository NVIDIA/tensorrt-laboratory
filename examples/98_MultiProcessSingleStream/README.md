# MPS Examples

`run_throughput_test ncopies batch_size engine_file MPS/NOMPS`

V100 - 16GB - DGX-1V

Processes | MPS | FPS | Batch | Model
--------- | --- | --- | ----- | -----
1 | N | 383 | 1 | RN50
8 | N | 365 | 1 | RN50
8 | Y | 929 | 1 | RN50

```
root@dgx11:/work/src/Examples/98_MultiProcessSingleStream# ./run_throughput_test 8 1 /work/models/ResNet-50-b1-fp32.engine MPS
starting 8 inference services
starting load balancer
load balancing over ports:  ['50051', '50052', '50053', '50054', '50055', '50056', '50057', '50058']
running test client
1000 requests in 1.07632seconds; inf/sec: 929.095

root@dgx11:/work/src/Examples/98_MultiProcessSingleStream# ./run_throughput_test 8 1 /work/models/ResNet-50-b1-fp32.engine NOMPS
starting 8 inference services
starting load balancer
load balancing over ports:  ['50051', '50052', '50053', '50054', '50055', '50056', '50057', '50058']
running test client
1000 requests in 2.74228seconds; inf/sec: 364.66

root@dgx11:/work/src/Examples/98_MultiProcessSingleStream# ./run_throughput_test 1 1 /work/models/ResNet-50-b1-fp32.engine NOMPS
starting 1 inference services
starting load balancer
load balancing over ports:  ['50051']
running test client
1000 requests in 2.60915seconds; inf/sec: 383.267
```
