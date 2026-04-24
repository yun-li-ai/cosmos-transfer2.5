"""Lilypad entrypoint for Cosmos Transfer 2.5 WFM inference."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import boto3
import ray
from lilypad.public.sdk_py.cached_file_access.boto import get_readonly_boto_client

logger = logging.getLogger(__name__)


def _plain_oci_client():
    # Used for LIST (assets) and PUT (outputs) — AIStore denies both operations.
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _download_prefix(client, bucket: str, prefix: str, local_dir: Path) -> None:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):].lstrip("/")
            dest = local_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading s3://%s/%s -> %s", bucket, key, dest)
            client.download_file(bucket, key, str(dest))


def _upload_dir(client, local_dir: Path, bucket: str, prefix: str) -> None:
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix}/{path.relative_to(local_dir)}".lstrip("/")
        logger.info("Uploading %s -> s3://%s/%s", path, bucket, key)
        client.upload_file(str(path), bucket, key)


_HF_PREDICT2_MODEL = "models--nvidia--Cosmos-Predict2.5-2B"
_HF_TOKENIZER_REVISION = "6787e176dce74a101d922174a95dba29fa5f0c55"
# SHA-256 of the tokenizer.pth blob (from local HF cache filename).
_HF_TOKENIZER_BLOB_SHA256 = (
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
)


@ray.remote
def _run_inference_on_gpu(config: dict) -> None:
    """Runs on the GPU worker node where the actual GPUs are available."""
    import logging
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import boto3
    from lilypad.public.sdk_py.cached_file_access.boto import get_readonly_boto_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    num_gpus = config.get("num_gpus", 8)
    experiment = config["experiment"]
    spec_json = config.get("spec_json", "multiview_spec.json")

    import botocore.config

    # OCI S3-compat requires payload signing for PUT operations and disables
    # checksum enforcement (which AWS SDK 2.x sends by default).
    _oci_config = botocore.config.Config(
        s3={"payload_signing_enabled": True},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    plain_client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=_oci_config,
    )
    cached_client = get_readonly_boto_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        assets_dir = work / "assets"
        checkpoint_dir = work / "checkpoints"
        output_dir = work / "outputs"

        logger.info("Downloading assets from s3://%s/%s", config["assets_bucket"], config["assets_prefix"])
        paginator = plain_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config["assets_bucket"], Prefix=config["assets_prefix"]):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(config["assets_prefix"]):].lstrip("/")
                dest = assets_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading s3://%s/%s -> %s", config["assets_bucket"], key, dest)
                plain_client.download_file(config["assets_bucket"], key, str(dest))

        checkpoint_dir.mkdir(parents=True)
        ckpt_local = checkpoint_dir / "model_ema_bf16.pt"
        logger.info("Downloading checkpoint from s3://%s/%s", config["checkpoint_bucket"], config["checkpoint_key"])
        cached_client.download_file(config["checkpoint_bucket"], config["checkpoint_key"], str(ckpt_local))

        # Pre-populate the HF hub cache with the Wan2.1 VAE tokenizer so
        # checkpoint_db.py finds it without hitting HuggingFace. The HF hub
        # uses a blobs/<sha256> + snapshots/<rev>/file -> ../../blobs/<sha256>
        # layout; placing only the file at the snapshot path is insufficient.
        hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        model_cache = hf_cache_dir / _HF_PREDICT2_MODEL
        blob_path = model_cache / "blobs" / _HF_TOKENIZER_BLOB_SHA256
        snapshot_dir = model_cache / "snapshots" / _HF_TOKENIZER_REVISION
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading VAE tokenizer from s3://%s/%s",
            config["tokenizer_bucket"],
            config["tokenizer_key"],
        )
        cached_client.download_file(config["tokenizer_bucket"], config["tokenizer_key"], str(blob_path))

        # Symlink: snapshots/<rev>/tokenizer.pth -> ../../blobs/<sha256>
        symlink_path = snapshot_dir / "tokenizer.pth"
        if not symlink_path.exists():
            symlink_path.symlink_to(Path("../../blobs") / _HF_TOKENIZER_BLOB_SHA256)

        # refs/main so snapshot_download resolves the default revision.
        refs_dir = model_cache / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(_HF_TOKENIZER_REVISION)

        # Propagate HF_TOKEN so the Qwen/Cosmos-Reason1-7B text encoder can
        # download from HuggingFace. Do NOT set HF_HUB_OFFLINE globally — it
        # would also block the Reason1-7B download.
        if "HF_TOKEN" in os.environ:
            logger.info("HF_TOKEN found; HuggingFace downloads will be authenticated.")
        else:
            logger.warning("HF_TOKEN not set; gated HuggingFace models may fail to download.")

        output_dir.mkdir(parents=True)

        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--master_port=12341",
            "-m", "examples.multiview",
            "-i", str(assets_dir / spec_json),
            "-o", str(output_dir),
            "--checkpoint_path", str(ckpt_local),
            "--experiment", experiment,
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            # Upload console.log (written by multiview init) before raising so
            # the traceback is retrievable from OCI even after the job fails.
            console_log = output_dir / "console.log"
            if console_log.exists():
                debug_key = f"{config['output_prefix']}/_debug/console.log"
                try:
                    plain_client.upload_file(str(console_log), config["output_bucket"], debug_key)
                    logger.info("Uploaded console.log to s3://%s/%s for debugging", config["output_bucket"], debug_key)
                except Exception as upload_err:
                    logger.warning("Could not upload console.log: %s", upload_err)
            raise RuntimeError(f"torchrun exited with code {result.returncode}")
        logger.info("torchrun finished successfully")

        output_files = [p for p in sorted(output_dir.rglob("*")) if p.is_file()]
        logger.info("Uploading %d output files to s3://%s/%s", len(output_files), config["output_bucket"], config["output_prefix"])
        for path in output_files:
            key = f"{config['output_prefix']}/{path.relative_to(output_dir)}".lstrip("/")
            logger.info("Uploading %s -> s3://%s/%s", path, config["output_bucket"], key)
            plain_client.upload_file(str(path), config["output_bucket"], key)
        logger.info("Upload complete")


def run(config: dict) -> None:
    """Lilypad entrypoint for Cosmos Transfer 2.5 multiview inference.

    Config keys:
        assets_bucket:      OCI bucket containing input assets
        assets_prefix:      prefix under which the assets/ tree is stored
        checkpoint_bucket:  OCI bucket containing the model checkpoint
        checkpoint_key:     full object key for model_ema_bf16.pt
        output_bucket:      OCI bucket to upload inference outputs to
        output_prefix:      prefix under which outputs will be written
        experiment:         --experiment arg passed to examples.multiview
        spec_json:          path relative to assets root to the spec JSON
                            (default: multiview_spec.json)
        num_gpus:           number of GPUs to use (default: 8)
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    num_gpus = config.get("num_gpus", 8)

    # Connect to the Ray cluster that Lilypad already set up. The head node
    # (where this runs) is CPU-only; GPUs live on the worker nodes. We dispatch
    # the heavy work there via a remote task that requests all GPUs.
    ray.init(address="auto")

    logger.info("Dispatching inference to GPU worker (num_gpus=%d)", num_gpus)
    ref = _run_inference_on_gpu.options(num_gpus=num_gpus).remote(config)
    ray.get(ref)
    logger.info("Inference complete.")
