# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Callback to upload training checkpoints to Weights & Biases as artifacts."""

import os

import wandb

from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.callback import Callback


class WandbCheckpointUploadCallback(Callback):
    """Uploads each saved checkpoint directory to wandb as a model artifact.

    Only runs when checkpoints are saved to local disk (not S3-only). Enable by
    adding this callback to trainer.callbacks (e.g. via config override).
    """

    def __init__(self, artifact_type: str = "model", aliases: str | None = None) -> None:
        super().__init__()
        self.artifact_type = artifact_type
        self.aliases = aliases  # e.g. "latest" to tag the most recent checkpoint.

    @distributed.rank0_only
    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        if not wandb.run:
            return
        # Checkpoints are only on disk when save_to_object_store is disabled.
        if getattr(self.config.checkpoint.save_to_object_store, "enabled", True):
            log.info("wandb_ckpt: Skipping upload (checkpoints are saved to object store only).")
            return
        ckpt_dir = os.path.join(self.config.job.path_local, "checkpoints", f"iter_{iteration:09}")
        if not os.path.isdir(ckpt_dir):
            log.warning(f"wandb_ckpt: Checkpoint dir not found {ckpt_dir}, skipping upload.")
            return
        name = f"checkpoint-iter-{iteration}"
        artifact = wandb.Artifact(name=name, type=self.artifact_type)
        artifact.add_dir(ckpt_dir, name=os.path.basename(ckpt_dir))
        try:
            wandb.log_artifact(artifact, aliases=[self.aliases] if self.aliases else None)
            log.info(f"wandb_ckpt: Logged artifact {name} to run {wandb.run.id}.")
        except Exception as e:
            log.warning(f"wandb_ckpt: Failed to log artifact: {e}.")
