FROM python:3.9.15-bullseye
# FROM nvcr.io/nvidia/pytorch:21.04-py3
FROM pytorch/torchserve:0.6.0-cpu

WORKDIR /app

COPY handler.py model.py utils.py config.properties index_to_classes.json requirements.txt /app/

USER root

RUN mkdir Dataset model_dir model_store serve

ADD Dataset /app/Dataset

ADD model_dir/model2.ckpt /app/model_dir

ADD serve /app/serve

RUN pip3 install -r requirements.txt

RUN torch-model-archiver \
    --model-name=flow-gated-network \
    --version=1.0 \
    --model-file=model_dir/model2.ckpt \
    --handler=handler.py \
    --export-path=model_store \
    --extra-files=index_to_classes.json

CMD ["torchserve", \
    "--start", \
    "--ncs", \
    "--model-store=/app/model_store", \
    "--ts-config=/app/config.properties", \
    "--models", \
    "fgn=flow-gated-network.mar"]
