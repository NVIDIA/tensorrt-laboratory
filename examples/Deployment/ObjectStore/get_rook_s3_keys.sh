#!/bin/bash
objstore=trtlab-s3
user=trtlab
echo -n export AWS_ACCESS_KEY_ID=
kubectl -n rook-ceph get secret rook-ceph-object-user-${objstore}-${user} -o yaml | grep AccessKey | awk '{print $2}' | base64 --decode
echo
echo -n export AWS_SECRET_ACCESS_KEY=
kubectl -n rook-ceph get secret rook-ceph-object-user-${objstore}-${user} -o yaml | grep SecretKey | awk '{print $2}' | base64 --decode
echo
echo export AWS_ENDPOINT_URL=http://s3.trt.lab
echo
