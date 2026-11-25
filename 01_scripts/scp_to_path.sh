#!/bin/bash
export SSH_ASKPASS_REQUIRE=force
export SSH_ASKPASS="$(pwd)/askpass.sh"
src="$1"
dst="$2"
scp -o StrictHostKeyChecking=no -P 36551 "$src" "root@connect.nmb2.seetacloud.com:$dst" < /dev/null
