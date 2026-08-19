# Self-contained build of the Task 3 submission.
#
# The submitted image was layered on a base built earlier, which is not reproducible as
# written: it pip-installed torch 2.5.1 from the cu121 index, and torch 2.5.1 pins
# nvidia-cudnn-cu12==9.1.0.70, which has since been withdrawn from that index. Taking torch
# from the official PyTorch image instead gives the same torch/CUDA/cuDNN combination without
# depending on a withdrawn wheel.
#
# The AssemblyNet baseline is not vendored here. Clone it into the build context first:
#   git clone https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline
# build.sh does this automatically.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Grand Challenge runs the algorithm as a non-root user.
RUN useradd --create-home --uid 999 --shell /bin/bash algorithm

WORKDIR /opt/algorithm
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=algorithm:algorithm PENGWIN2026_Task3_Reduction_Baseline/ \
     /opt/algorithm/external/PENGWIN2026_Task3_Reduction_Baseline/
COPY --chown=algorithm:algorithm scripts/ /opt/algorithm/scripts/task3/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/process.py

RUN mkdir -p /input /output && chown -R algorithm:algorithm /opt/algorithm /output

USER algorithm
# Five identity poses with lopsided weights must come back as identity, and a five-member
# call must reject four weights. Fails the build rather than the submission.
RUN python -c "import sys, numpy as np; sys.path.insert(0,'/opt/algorithm/scripts/task3'); \
from pose_ensemble import combine_predictions; \
p=[[{'fragment_id':'1','transformation':np.eye(4).tolist()}] for _ in range(5)]; \
r=combine_predictions(p, {'1': np.zeros(3)}, [0.6,0.0,0.1333,0.1333,0.1334], [0.2]*5); \
assert np.allclose(np.asarray(r[0]['transformation']), np.eye(4)), 'identity failed'; \
ok=False; \
exec('try:\n combine_predictions(p, {\"1\": np.zeros(3)}, [0.25]*4, [0.2]*5)\nexcept ValueError:\n ok=True'); \
assert ok, 'weight-count check missing'; print('pose_ensemble ok (5 members)')"

ENTRYPOINT ["python", "/opt/algorithm/process.py"]
