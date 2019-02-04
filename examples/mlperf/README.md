

```
git clone ssh://git@gitlab-master.nvidia.com:12051/rolson/tensorrt-laboratory.git
cd tensorrt-laboratory
make
./devel.sh
./build.sh
cd models
./mps_builder 50
./setup.py
# exit or ctrl+d to escape the MPS shell
exit
cd /work/build/examples/mlperf
make
./mlperf-infer.x
```
