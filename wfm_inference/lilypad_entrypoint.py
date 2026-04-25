"""Lilypad entrypoint for Cosmos Transfer 2.5 WFM inference."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import boto3
import botocore.config
import ray
from lilypad.public.sdk_py.cached_file_access.boto import get_readonly_boto_client

logger = logging.getLogger(__name__)

# OCI S3-compat requires payload signing for PUT and disables the default
# AWS SDK v2 checksum headers that OCI doesn't support.
_OCI_BOTO_CONFIG = botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)


def _plain_oci_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=_OCI_BOTO_CONFIG,
    )


def _remap_hf_snapshot(
    hf_cache_dir: Path,
    repo: str,
    expected_revision: str,
    logger: "logging.Logger",
) -> None:
    """Copy snapshots/<actual_rev>/ to snapshots/<expected_rev>/ when they differ.

    The OCI cache may have been staged at a different commit than what
    checkpoint_db.py requests. Since file content is identical, we can just
    alias the snapshot directory. HF hub with HF_HUB_OFFLINE=1 looks up files
    by snapshot path, not by blob hash, so this is sufficient.
    """
    import shutil

    model_dir = hf_cache_dir / ("models--" + repo.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    if not refs_main.exists():
        logger.warning("_remap_hf_snapshot: refs/main not found for %s, skipping", repo)
        return

    actual_revision = refs_main.read_text().strip()
    if actual_revision == expected_revision:
        return

    actual_snapshot = model_dir / "snapshots" / actual_revision
    expected_snapshot = model_dir / "snapshots" / expected_revision

    if not actual_snapshot.exists():
        logger.warning("_remap_hf_snapshot: snapshot %s not found for %s, skipping", actual_revision[:8], repo)
        return

    if not expected_snapshot.exists():
        shutil.copytree(str(actual_snapshot), str(expected_snapshot), symlinks=True)
        logger.info("Remapped %s: %s -> %s", repo, actual_revision[:8], expected_revision[:8])


@ray.remote
def _run_inference_on_gpu(config: dict) -> None:
    """Runs on the GPU worker node where the actual GPUs are available."""
    import logging
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import boto3
    import botocore.config
    from lilypad.public.sdk_py.cached_file_access.boto import get_readonly_boto_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    num_gpus = config.get("num_gpus", 8)
    experiment = config["experiment"]
    spec_json = config.get("spec_json", "multiview_spec.json")

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

        # Populate HF hub cache from OCI so checkpoint_db.py finds all
        # auxiliary models (VAE tokenizer, Cosmos-Reason1-7B text encoder)
        # without hitting HuggingFace. The OCI cache stores real files at
        # snapshots/<rev>/<filename> — HF hub finds them directly.
        hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        hf_cache_bucket = config["hf_cache_bucket"]
        hf_cache_prefix = config["hf_cache_prefix"].rstrip("/")
        logger.info("Downloading HF model cache from s3://%s/%s", hf_cache_bucket, hf_cache_prefix)
        for page in paginator.paginate(Bucket=hf_cache_bucket, Prefix=hf_cache_prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(hf_cache_prefix):].lstrip("/")
                dest = hf_cache_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading %s -> %s", key, dest)
                cached_client.download_file(hf_cache_bucket, key, str(dest))

        # checkpoint_db.py hardcodes specific git revisions that may differ from
        # whatever revision was current when we staged the OCI cache. After
        # downloading, create snapshot aliases so HF hub finds the expected paths.
        _remap_hf_snapshot(
            hf_cache_dir,
            repo="nvidia/Cosmos-Predict2.5-2B",
            expected_revision="6787e176dce74a101d922174a95dba29fa5f0c55",
            logger=logger,
        )
        _remap_hf_snapshot(
            hf_cache_dir,
            repo="nvidia/Cosmos-Reason1-7B",
            expected_revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
            logger=logger,
        )

        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("HF cache populated; offline mode enabled")

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
            # Guardrail model (nvidia/Cosmos-Guardrail1) is not staged in OCI;
            # we don't need content safety checks for internal inference.
            "--disable-guardrails",
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            # Upload console.log before raising so the traceback is readable from OCI.
            console_log = output_dir / "console.log"
            if console_log.exists():
                debug_key = f"{config['output_prefix']}/_debug/console.log"
                try:
                    plain_client.upload_file(str(console_log), config["output_bucket"], debug_key)
                    logger.info("Uploaded console.log to s3://%s/%s", config["output_bucket"], debug_key)
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
        hf_cache_bucket:    OCI bucket containing the pre-staged HF model cache
        hf_cache_prefix:    prefix under which the HF cache tree is stored
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
