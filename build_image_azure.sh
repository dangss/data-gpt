#!/bin/bash
sh compress_datasets.sh #=> run this first
az acr build --image digpt:v1 --registry crdigpt --file Dockerfile .