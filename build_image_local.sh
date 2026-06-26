#!/bin/bash
docker image prune -a --force && \
docker build -t chatdigpt:v1 . && \
docker save chatdigpt:v1 | gzip > chat_digpt_v1.tar.gz

#--no-cache --progress=plain
#docker tag digpt:v1 crdigpt.azurecr.io/digpt:v1
# docker push crdigpt.azurecr.io/digpt:v1