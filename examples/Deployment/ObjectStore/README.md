# Object Store

In the Image Service example, the ImageClient separates out an inference request
into two components: 
- 1) a bulk data transfers to a backend store, 
- 2) and a gRPC requests that essentially captures the metadata for the type of
  operation and location of the image in the backend store.

The implemenation of this concept uses an S3-compatible Object Store. The
samples should work equally well on AWS S3 or via Rook's S3 Object Store
implementation running locally on our Kubernetes cluster. For more details on
how we installed Kubernetes and Rook, see the [NVIDIA DeepOps Project](https://github.com/nvidia/deepops).

This folder contains some basic configuration files and scripts for preparing the
ObjectStore for our Image Service.

## Rook + Kubernetes

You may need to modify some of the configuration files for your cluster.

- `rook-s3.yml` options 
  - requires 3 unique hosts with bluestore backed OSDs
  - creates a `trtlab` user

If you modify the name of the ObjectStore (`trtlab-s3`) and or the username (`trtlab`),
be aware the `get_rook_s3_keys.sh` needs to be modified:

```
rook-ceph-object-user-<object-store-name>-<username>
```

Similarly, the endpoint to host the storage is `s3.trt.lab`. If you change this,
you will need to modify `get_rook_s3_keys.sh` to output the proper
`AWS_ENDPOINT_URL`. You will also need to modify the ingress examples.

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