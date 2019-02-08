#!/bin/bash 

cleanup() {
  kill $(jobs -p) ||:
}
trap "cleanup" EXIT SIGINT SIGTERM

./nvrpc-bidirectional-server.x --ip_port="0.0.0.0:5555" &

f=$(mktmp)
cat <<EOF > $f
PS1='nvRPC Bidirectional: '

go() { ./nvrpc-bidirectional-client.x --hostname="localhost:5555" --count=${1:-100} }
EOF

ps aux

echo
echo 'Try ./nvrpc-bidirectional-client.x --hostname="localhost:5555" --count=100'
bash --rcfile <(echo "PS1='nvRPC Bidirectional: '")
