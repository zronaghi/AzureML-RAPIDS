FROM rapidsai/rapidsai:0.11-cuda10.0-runtime-ubuntu18.04-py3.7
MAINTAINER zronaghi@nvidia.com

RUN source activate rapids && \
    pip install --upgrade azureml-sdk && \
    pip install azureml-widgets
