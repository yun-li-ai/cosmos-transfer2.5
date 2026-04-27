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

# OCI S3-compat requires payload signing and disables the default AWS SDK v4
# checksum headers that OCI doesn't support.
_OCI_BOTO_CONFIG = botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)

# Persistent directory on the worker node for shared resources that survive
# across jobs in a batch (checkpoint, HF cache). Lives outside tempdir so it
# is not cleaned up between jobs.
_WORKER_CACHE_DIR = Path("/tmp/wfm_worker_cache")


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


def _download_checkpoint(
    cached_client: "boto3.client",
    bucket: str,
    key: str,
    logger: "logging.Logger",
) -> Path:
    """Download model checkpoint to a persistent cache; skip if already present."""
    dest = _WORKER_CACHE_DIR / "checkpoints" / bucket / key.lstrip("/")
    if dest.exists():
        logger.info("Checkpoint already cached at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading checkpoint s3://%s/%s -> %s", bucket, key, dest)
    cached_client.download_file(bucket, key, str(dest))
    return dest


def _setup_hf_cache(
    plain_client: "boto3.client",
    cached_client: "boto3.client",
    hf_cache_bucket: str,
    hf_cache_prefix: str,
    hf_cache_dir: Path,
    logger: "logging.Logger",
) -> None:
    """Download HF model cache from OCI; skip if expected snapshots already exist."""
    predict2b_snapshot = (
        hf_cache_dir
        / "models--nvidia--Cosmos-Predict2.5-2B"
        / "snapshots"
        / "6787e176dce74a101d922174a95dba29fa5f0c55"
    )
    reason1_snapshot = (
        hf_cache_dir
        / "models--nvidia--Cosmos-Reason1-7B"
        / "snapshots"
        / "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
    )
    if predict2b_snapshot.exists() and reason1_snapshot.exists():
        logger.info("HF cache already populated, skipping download")
        return

    hf_cache_prefix = hf_cache_prefix.rstrip("/")
    logger.info("Downloading HF model cache from s3://%s/%s", hf_cache_bucket, hf_cache_prefix)
    paginator = plain_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=hf_cache_bucket, Prefix=hf_cache_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(hf_cache_prefix):].lstrip("/")
            dest = hf_cache_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            cached_client.download_file(hf_cache_bucket, key, str(dest))

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


@ray.remote
def _run_batch_on_gpu(base_config: dict, jobs: list[dict]) -> None:
    """Runs on the GPU worker. Downloads shared resources once, then runs all jobs."""
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

    hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

    # Download shared resources once for the whole batch.
    checkpoint_path = _download_checkpoint(
        cached_client,
        base_config["checkpoint_bucket"],
        base_config["checkpoint_key"],
        logger,
    )
    _setup_hf_cache(
        plain_client,
        cached_client,
        base_config["hf_cache_bucket"],
        base_config["hf_cache_prefix"],
        hf_cache_dir,
        logger,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("Shared resources ready; running %d job(s)", len(jobs))

    num_gpus = base_config.get("num_gpus", 8)
    experiment = base_config["experiment"]

    for i, job in enumerate(jobs):
        input_bucket = job["input_bucket"]
        input_prefix = job["input_prefix"]
        output_bucket = job["output_bucket"]
        output_prefix = job["output_prefix"]
        spec_json = job.get("spec_json", "multiview_spec.json")

        logger.info("Job %d/%d: s3://%s/%s -> s3://%s/%s",
                    i + 1, len(jobs), input_bucket, input_prefix, output_bucket, output_prefix)

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            assets_dir = work / "assets"
            output_dir = work / "outputs"
            output_dir.mkdir(parents=True)

            paginator = plain_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=input_bucket, Prefix=input_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative = key[len(input_prefix):].lstrip("/")
                    dest = assets_dir / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    plain_client.download_file(input_bucket, key, str(dest))

            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                "--master_port=12341",
                "-m", "examples.multiview",
                "-i", str(assets_dir / spec_json),
                "-o", str(output_dir),
                "--checkpoint_path", str(checkpoint_path),
                "--experiment", experiment,
                # Guardrail model (nvidia/Cosmos-Guardrail1) is not staged in OCI;
                # not needed for internal inference on controlled driving data.
                "--disable-guardrails",
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=False)

            if result.returncode != 0:
                console_log = output_dir / "console.log"
                if console_log.exists():
                    debug_key = f"{output_prefix}/_debug/console.log"
                    try:
                        plain_client.upload_file(str(console_log), output_bucket, debug_key)
                        logger.info("Uploaded console.log to s3://%s/%s", output_bucket, debug_key)
                    except Exception as upload_err:
                        logger.warning("Could not upload console.log: %s", upload_err)
                raise RuntimeError(f"Job {i + 1}/{len(jobs)} torchrun exited with code {result.returncode}")

            logger.info("Job %d/%d torchrun finished successfully", i + 1, len(jobs))

            output_files = [p for p in sorted(output_dir.rglob("*")) if p.is_file()]
            logger.info("Uploading %d file(s) to s3://%s/%s", len(output_files), output_bucket, output_prefix)
            for path in output_files:
                key = f"{output_prefix}/{path.relative_to(output_dir)}".lstrip("/")
                plain_client.upload_file(str(path), output_bucket, key)
            logger.info("Job %d/%d upload complete", i + 1, len(jobs))


def run(config: dict) -> None:
    """Lilypad entrypoint for Cosmos Transfer 2.5 multiview inference.

    Accepts either a single job (flat config) or a batch (jobs list). Shared
    resources (model checkpoint, HF model cache) are downloaded once per batch
    and reused across all jobs.

    Base config keys (shared across all jobs):
        checkpoint_bucket:  OCI bucket containing the model checkpoint
        checkpoint_key:     full object key for model_ema_bf16.pt
        hf_cache_bucket:    OCI bucket containing the pre-staged HF model cache
        hf_cache_prefix:    prefix under which the HF cache tree is stored
        experiment:         --experiment arg passed to examples.multiview
        num_gpus:           number of GPUs to use (default: 8)

    Per-job keys (under the jobs list, or at top level for a single job):
        input_bucket:      OCI bucket containing input assets
        input_prefix:      prefix under which the assets/ tree is stored
        output_bucket:      OCI bucket to upload inference outputs to
        output_prefix:      prefix under which outputs will be written
        spec_json:          spec file path relative to assets root
                            (default: multiview_spec.json)
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    num_gpus = config.get("num_gpus", 8)

    if "jobs" in config:
        jobs = config["jobs"]
        base_config = {k: v for k, v in config.items() if k != "jobs"}
    else:
        # Single-job flat format for backward compatibility.
        jobs = [{
            "input_bucket": config["input_bucket"],
            "input_prefix": config["input_prefix"],
            "output_bucket": config["output_bucket"],
            "output_prefix": config["output_prefix"],
            "spec_json": config.get("spec_json", "multiview_spec.json"),
        }]
        base_config = config

    ray.init(address="auto")
    logger.info("Dispatching batch of %d job(s) to GPU worker (num_gpus=%d)", len(jobs), num_gpus)
    ref = _run_batch_on_gpu.options(num_gpus=num_gpus).remote(base_config, jobs)
    ray.get(ref)
    logger.info("Batch complete.")
