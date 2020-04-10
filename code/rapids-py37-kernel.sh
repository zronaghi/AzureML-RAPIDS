#!/bin/bash
set -e

mkdir -p /home/azureuser/anaconda3/envs/rapids_py37
wget -q https://drobison.blob.core.windows.net/drobison-gtc-2020/rapids_py37.tar.gz
tar -xzf rapids_py37.tar.gz -C /home/azureuser/anaconda3/envs/rapids_py37

source activate rapids_py37

ipython kernel install --user --name=rapids_py37
