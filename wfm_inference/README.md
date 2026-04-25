# WFM Inference — Lilypad Entrypoint

Runs Cosmos Transfer 2.5 multiview inference on Lilypad generic workloads.
The workload config lives at `adp/services/wfm/workload_configs/cosmos_transfer_inference.yaml`
in the applied3 repo.

## Architecture

Lilypad generic workloads start a Ray cluster with two node types:

- **Head node** (`InstanceTypeVMStandardE5Flex`, CPU-only) — runs `lilypad_entrypoint.run()`.
- **GPU worker nodes** (A100×8) — where all the heavy work actually happens.

`run()` calls `_run_inference_on_gpu.remote(config)` and blocks on `ray.get()`. All
download, inference, and upload logic lives inside the `@ray.remote` function so it
executes on a GPU worker, not the CPU head.

## Usage

Export OCI credentials, then launch:

```bash
export AWS_ACCESS_KEY_ID=<oci-access-key>
export AWS_SECRET_ACCESS_KEY=<oci-secret-key>
lilypad workload launch adp/services/wfm/workload_configs/cosmos_transfer_inference.yaml
```

The workload config contains a base `entrypoint_fn_config` that the WFM gRPC service
will override per-job at submission time. For manual test runs the defaults point at
`sensor-sim-wfm/inputs/multiview_example`.

## Config Keys

| Key | Description |
|-----|-------------|
| `assets_bucket` / `assets_prefix` | OCI location of the input asset tree (spec JSON + camera frames) |
| `checkpoint_bucket` / `checkpoint_key` | OCI path to `model_ema_bf16.pt` |
| `hf_cache_bucket` / `hf_cache_prefix` | OCI prefix holding the pre-staged HuggingFace model cache (see below) |
| `output_bucket` / `output_prefix` | OCI destination for generated video files |
| `experiment` | `--experiment` arg passed to `examples.multiview` |
| `spec_json` | Spec file path relative to the assets root (default: `multiview_spec.json`) |
| `num_gpus` | GPUs to claim on the worker (default: 8) |

## Building and Pushing the Docker Image

All inference code is baked into the image at build time (`STANDALONE=true`).

```bash
cd /home/yun/cosmos-transfer2.5
docker build -f Dockerfile \
  --build-arg CUDA_NAME=cu128 \
  --build-arg STANDALONE=true \
  -t us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_transfer2.5_v<VERSION> .

docker push us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_transfer2.5_v<VERSION>
```

Then update `docker_image` in `cosmos_transfer_inference.yaml` to match.

Most layers are shared with the previous tag, so rebuilds are fast (under a minute when
only Python files changed).

## OCI S3-compat Gotcha

OCI's S3-compatible API requires payload signing and does not accept the default AWS SDK v4
checksum headers. Any boto3 client used for PUT/LIST against OCI must use:

```python
botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)
```

There are two clients in the worker:
- **`plain_client`** — direct OCI client (above config). Used for LIST, PUT, and
  downloads where AIStore caching is not desired.
- **`cached_client`** (`get_readonly_boto_client()`) — routes GETs through the AIStore
  cross-region cache at the Chicago edge. Used for checkpoint and HF cache downloads
  to avoid repeated cross-region transfer costs.

## Pre-staging the HuggingFace Model Cache

The inference pipeline (`checkpoint_db.py`) downloads two auxiliary HF models at startup:

| Model | HF repo | Revision expected by checkpoint_db |
|-------|---------|-------------------------------------|
| VAE tokenizer | `nvidia/Cosmos-Predict2.5-2B` | `6787e176dce74a101d922174a95dba29fa5f0c55` |
| Text encoder | `nvidia/Cosmos-Reason1-7B` | `3210bec0495fdc7a8d3dbb8d58da5711eab4b423` |

To avoid requiring `HF_TOKEN` at runtime, these are pre-staged in OCI under
`sensor-sim-wfm/checkpoints/hf-cache/` as a standard HF hub cache tree:

```
models--nvidia--Cosmos-Predict2.5-2B/
  refs/main
  snapshots/<rev>/tokenizer.pth
  blobs/...

models--nvidia--Cosmos-Reason1-7B/
  refs/main
  snapshots/<rev>/<all model files>
  blobs/...
```

Upload with `aws s3 sync` pointed at `~/.cache/huggingface/hub/` after downloading the
models locally. Requires these env var overrides to fix OCI upload errors:

```bash
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
aws s3 sync ~/.cache/huggingface/hub/ \
  s3://sensor-sim-wfm/checkpoints/hf-cache/ \
  --endpoint-url https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com
```

### Revision mismatch fix

`checkpoint_db.py` hardcodes specific git revisions. When those revisions were staged,
the HF repo `main` branch pointed at a different commit, so the OCI cache contains
snapshots at the wrong revision paths.

`_remap_hf_snapshot()` handles this after download: it reads `refs/main` from the local
cache, copies `snapshots/<actual>/` to `snapshots/<expected>/` if they differ, then sets
`HF_HUB_OFFLINE=1`. From that point, `uvx hf download --revision <expected>` finds the
files locally and never contacts HuggingFace.

If `checkpoint_db.py` is updated to a new revision, update the `expected_revision`
arguments in `_remap_hf_snapshot` calls and re-stage the OCI cache if the model weights
actually changed.

## Content Safety Guardrails

`examples.multiview` enables guardrails by default, which would download
`nvidia/Cosmos-Guardrail1` (~7B LLaMA-based safety model) from HuggingFace at startup.
That model is not staged in OCI and is not needed for internal inference on controlled
driving data.

Guardrails are disabled by passing `--disable-guardrails` to the torchrun command.
If guardrail checks are ever needed, stage `nvidia/Cosmos-Guardrail1` at revision
`d6d4bfa899a71454a700907664f3e88f503950cf` in OCI and remove that flag (and add a
`_remap_hf_snapshot` call for it).
