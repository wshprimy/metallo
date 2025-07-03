#!/bin/bash

VERSION=${1:-1.1.0}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
docker build \
    --network=host \
    --build-arg http_proxy= \
    --build-arg https_proxy= \
    --build-arg HTTP_PROXY= \
    --build-arg HTTPS_PROXY= \
    --build-arg no_proxy= \
    --build-arg NO_PROXY= \
    -t wangshihe/pytorch:$VERSION .
