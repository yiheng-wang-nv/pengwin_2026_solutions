# PENGWIN 2026 Task 1 — Grand Challenge algorithm container.
#
# Two-stage nnUNetv2 pipeline (anatomical + CSM) + post-processing.
# Model weights are NOT baked in — they are uploaded separately on Grand
# Challenge and extracted to /opt/ml/model at runtime (keeps the image small).
#
# Runtime contract (Grand Challenge):
#   - read CT from /input  (image interface slug: pelvic-facture-ct)
#   - write segmentation to /output (slug: pelvic-fracture-segmentation)
#   - no network access; /opt/ml/model is read-only with the model tarball
#   - GPU available (T4 16GB or A10G 24GB)

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    nnUNet_results=/opt/ml/model \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed

# nnUNet (and SimpleITK file IO) need a couple of system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (Grand Challenge runs containers as an unprivileged user).
RUN groupadd -r algorithm && useradd -r -g algorithm -m algorithm

WORKDIR /opt/algorithm
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY frac_to_instance.py inference.py ./

# Writable scratch for intermediate nnUNet files.
RUN mkdir -p /tmp/nnunet_raw /tmp/nnunet_preprocessed && chown -R algorithm:algorithm /tmp /opt/algorithm
USER algorithm

ENTRYPOINT ["python", "/opt/algorithm/inference.py"]
