# Object Store

In our ImageService Deploymenet example, our ImageClient separates our an
inference request into two components: a bulk data transfers to a backend store
and a gRPC requests that essentially captures the metadata for the type of
operation and location of the image in the backend store.

Our implemenation of this concept uses an S3-compatible Object Store. Our
samples should work equally well on AWS S3 or via Rook's S3 Object Store
implementation running locally on our Kubernetes cluster. For more details on
how we installed Kubernetes and Rook, see our [NVIDIA DeepOps Project](https://github.com/nvidia/deepops).

In this folder is some basic configuration files and scripts for preparing the
ObjectStore for our Image Service.

## Rook + Kubernetes

You may need to modify some of the configuration files for your cluster.

- `rook-s3.yml` 
  - requires 3 unique hosts with bluestore backed OSDs
  - creates a `trtlab` user

If you modify the name of the ObjectStore (`trtlab-s3`) and or the username,
be aware the `get_rook_s3_keys.sh` needs to be modified:

```
rook-ceph-object-user-<object-store-name>-<username>
```

Similarly, the endpoint to host the storage is `s3.trt.lab`.  If you change this,
you will need to modify `get_rook_s3_keys.sh` to output the proper `AWS_ENDPOINT_URL`

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