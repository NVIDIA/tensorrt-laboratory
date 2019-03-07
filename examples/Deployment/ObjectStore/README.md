# Object Store

In the Image Service example, the ImageClient separates out an inference request
into two components: 
- a bulk data transfers to a backend store, 
- a gRPC request that contains the details of the request (model, file_handle, etc.)

To implement this concept, we will use an S3-compatible Object Store. The
example should work equally well on AWS S3 or via Rook's S3 CephObjectStore
implementation running locally a Kubernetes cluster. For more details on
how Kubernetes and Rook were installed, see the [NVIDIA DeepOps Project](https://github.com/nvidia/deepops).

This folder contains some basic configuration files and scripts for preparing the
ObjectStore for our Image Service.

## AWS S3

Simply sent your AWS API configuration or export the following environment variables.

```
export AWS_ACCESS_KEY_ID=<your-aws-access-key>
export AWS_SECRET_ACCESS_KEY=<your-super-secret-access-key>
```

## Rook + Kubernetes

You will need to modify some of the configuration files for your cluster.

- `rook-s3.yml` options:
  - requires 3 unique hosts with bluestore backed OSDs
  - creates a `trtlab` user

If you modify the name of the ObjectStore (`trtlab-s3`) and or the username (`trtlab`),
be aware the `get_rook_s3_keys.sh` needs to be modified.

Similarly, the examples uses `s3.trt.lab` as the endpoint on which the storage
is hosted. If you change this, you will need to modify `get_rook_s3_keys.sh` to
output the proper `AWS_ENDPOINT_URL`. You will also need to modify the ingress
examples with the proper hostname.

```
kubectl apply -f rook-s3.yml
kubectl apply -f ingress-nginx.yml
```

### Setup your environment
```
eval $(./get_rook_s3_keys.sh)
```

### Prepare your Image bucket

Note: you will need to have python3 and boto installed.  This does not have be done
inside the container.

```
python3 create_buckets.py
```

## TODOs

- [ ] Export S3 keys to a Kubernetes Secret
- [ ] Scripts for bucket maintenance: probably some k8s CronJobs
- [ ] Update Istio ingress example