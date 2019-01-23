

```
git clone ssh://git@gitlab-master.nvidia.com:12051/rolson/tensorrt-playground.git
cd tensorrt-playground
make
./devel.sh
./build.sh
cd models
./mps_builder 50
./setup.py
cd /work/build/examples/mlperf
make
./mlperf-infer.x
```
