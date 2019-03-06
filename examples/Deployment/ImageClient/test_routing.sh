#!/bin/bash

cleanup() {
  kill $(jobs -p) ||:
  echo quit | nvidia-cuda-mps-control > /dev/null 2>&1 ||:
}
trap "cleanup" EXIT SIGINT SIGTERM

exe=/work/build/examples/Deployment/ImageClient/test_image_service.x

$exe --hostname="model_a" --ip_port="0.0.0.0:51051" & #> /dev/null 2>&1 &
$exe --hostname="model_b" --ip_port="0.0.0.0:51052" & #> /dev/null 2>&1 &
$exe --hostname="general_pool" --ip_port="0.0.0.0:51053" & #> /dev/null 2>&1 &
envoy -c envoy_config.yaml > /dev/null 2>&1 &

wait-for-it.sh localhost:50050 --timeout=0 -- echo "Envoy on 50050 ready"
wait-for-it.sh localhost:51051 --timeout=0 -- echo "ModelA on 51051 ready"
wait-for-it.sh localhost:51052 --timeout=0 -- echo "ModelB on 51052 ready"
wait-for-it.sh localhost:51053 --timeout=0 -- echo "General Pool on 51053 ready"

python3 test_client.py

