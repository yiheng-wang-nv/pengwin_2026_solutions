# Self-contained build of the Task 2 submission.
#
# torch and torchvision come from the base image rather than from pip: torch 2.5.1 pins
# nvidia-cudnn-cu12==9.1.0.70, which has since been withdrawn from the cu121 index, so a
# pip install of torch at this version no longer resolves. requirements.txt therefore lists
# only the packages layered on top.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    nnUNet_results=/opt/ml/model \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Grand Challenge runs the algorithm as a non-root user.
RUN groupadd -r algorithm && useradd -r -g algorithm -m algorithm

WORKDIR /opt/algorithm
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY inference/ /opt/algorithm/inference/
COPY process.py /opt/algorithm/process.py

RUN mkdir -p /tmp/nnunet_raw /tmp/nnunet_preprocessed /input /output \
    && chown -R algorithm:algorithm /tmp /opt/algorithm /output

USER algorithm
ENV TILE_STEP=0.5
ENTRYPOINT ["python", "/opt/algorithm/process.py"]
