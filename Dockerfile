ARG ROCM_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${ROCM_IMAGE}

ARG CLINFUSION_REF=96dd208ea7cdfecd001a703e232aa607802a8fd6
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    HF_HOME=/opt/ClinFusion/cache/huggingface \
    TRANSFORMERS_CACHE=/opt/ClinFusion/cache/huggingface \
    ATTENTION_BACKEND=TORCH_SDPA \
    CLINFUSION_ATTN_IMPL=sdpa \
    CLINFUSION_DTYPE=fp16 \
    CLINFUSION_STRICT_3D=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN git clone https://github.com/alibaba-damo-academy/ClinFusion.git /opt/ClinFusion \
    && cd /opt/ClinFusion \
    && git checkout "${CLINFUSION_REF}"

COPY requirements-rocm.txt /tmp/requirements-rocm.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-rocm.txt

COPY scripts/patch_clinfusion.py /tmp/patch_clinfusion.py
RUN python /tmp/patch_clinfusion.py /opt/ClinFusion

COPY api /app/api
COPY scripts /app/scripts
RUN touch /app/api/__init__.py /app/scripts/__init__.py \
    && mkdir -p /opt/ClinFusion/cache/models /opt/ClinFusion/cache/3d_volume /data /tmp/clinfusion_uploads

ENV PYTHONPATH=/app:/opt/ClinFusion
WORKDIR /opt/ClinFusion
EXPOSE 8008
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8008"]
