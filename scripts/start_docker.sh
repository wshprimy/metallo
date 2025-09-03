#!/bin/bash

VERSION=${1:-1.1.0}

docker run -itd \
    --gpus all \
    --shm-size=512g \
    -v $(pwd):/workspace/metallo \
    -v /data/wangshihe/ramdisk:/ramdisk \
    --network=host \
    -e http_proxy=http://127.0.0.1:10809 \
    -e https_proxy=http://127.0.0.1:10809 \
    -e HTTP_PROXY=http://127.0.0.1:10809 \
    -e HTTPS_PROXY=http://127.0.0.1:10809 \
    --name wsh-m-1 \
    wangshihe/pytorch:$VERSION
