# Development with Minikube

## Install `minikube`, `kubectl`, and `helm`

This only need to be done one time, or periodically if you wish to upgrade.

```
sudo apt update && sudo apt install -y --no-install-recommends socat

curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/

curl -Lo kubectl https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl && chmod +x kubectl && sudo mv kubectl /usr/local/bin

curl https://raw.githubusercontent.com/kubernetes/helm/master/scripts/get > get_helm.sh
chmod 700 get_helm.sh
./get_helm.sh
```

Add the `coreos/prometheus-operator` repo:
```
helm repo add coreos https://s3-eu-west-1.amazonaws.com/coreos-charts/stable/
```

## Launch a Kubernetes Cluster

```
./bootstrap.sh
```

Check configurations:
```
kubectl get all
kubectl get all --all-namespaces
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPUs:.status.capacity.'nvidia\.com/gpu'
# last command should report the number of GPUs on your system
# this make take some time - coffee?!
```

