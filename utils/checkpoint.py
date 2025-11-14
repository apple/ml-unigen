#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import json
import shutil
from pathlib import Path
import components.core as core


logger = core.get_logger()

def save_checkpoint(model, config, accelerator, global_step, save_last=False, optimizer=None, scheduler=None):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    logger.info("==> Saving checkpoint  <==")
    unwrapped_model = accelerator.unwrap_model(model)
    if not save_last:
        accelerator.save_state(save_path)
    else:
        # elif accelerator.is_main_process:
        unwrapped_model.register_to_config(load_from_pretrained=True)
        unwrapped_model.llm.config.use_cache = True
        state_dict = accelerator.get_state_dict(model)
    
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        
    if accelerator.is_main_process:
        logger.info(f"Saved state to {save_path}")
        if optimizer is not None:
            logger.info("==> Saving optimizer <==")
            accelerator.save(optimizer.state_dict(),  save_path / "optimizer.pth.tar")
        if scheduler is not None:
            logger.info("==> Saving scheduler <==")
            accelerator.save(scheduler.state_dict(), save_path / "scheduler.pth.tar")
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))