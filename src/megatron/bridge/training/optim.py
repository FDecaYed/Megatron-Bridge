# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Union

from megatron.core.optimizer import (
    MegatronOptimizer,
    OptimizerConfig,
    ParamKey,
    ParamPredicate,
    get_megatron_optimizer,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler, ParamGroupOverride
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.training.config import SchedulerConfig


def _mark_qk_layernorm_params(model: Union[MegatronModule, list[MegatronModule]]) -> None:
    """Mark q_layernorm and k_layernorm parameters for Qwen3-Next weight decay handling.

    This function sets the `is_qk_layernorm` attribute on parameters whose names
    contain 'q_layernorm' or 'k_layernorm'. This allows the config_overrides predicate
    to identify these parameters without needing model-level changes.

    Args:
        model: The model or list of model chunks to process
    """
    model_list = model if isinstance(model, list) else [model]
    for model_chunk in model_list:
        for name, param in model_chunk.named_parameters():
            if "q_layernorm" in name or "k_layernorm" in name:
                param.is_qk_layernorm = True


def _build_config_overrides(
    scheduler_config: SchedulerConfig,
) -> Optional[Dict[ParamKey, ParamGroupOverride]]:
    """Build config overrides for weight decay based on scheduler configuration.

    This function creates parameter-specific overrides for weight decay behavior.
    By default, weight decay is skipped for bias parameters and 1D parameters.
    For Qwen3-Next models, weight decay is applied to q_layernorm and k_layernorm.

    Args:
        scheduler_config: Scheduler configuration containing weight decay settings

    Returns:
        Dictionary of ParamKey to ParamGroupOverride for the optimizer
    """
    config_overrides: Dict[ParamKey, ParamGroupOverride] = {}

    # Always skip weight decay for bias parameters
    bias_key = ParamKey(name="*.bias")
    config_overrides[bias_key] = ParamGroupOverride(wd_mult=0.0)

    if scheduler_config.no_weight_decay_cond_type == "qwen3_next":
        # Qwen3-Next applies weight decay to qk layernorm as a special case.
        # Create a predicate that skips weight decay for 1D params EXCEPT q/k_layernorm.
        # The is_qk_layernorm attribute is set by _mark_qk_layernorm_params() based on param names.
        def qwen3_no_wd_1d_cond(param):
            # Skip weight decay for 1D params that are NOT q/k_layernorm
            if len(param.shape) != 1:
                return False
            # If param has q/k layernorm marker, don't skip weight decay
            if getattr(param, "is_qk_layernorm", False):
                return False
            return True

        param_1d_except_qk_ln = ParamPredicate(
            name="param_len_1_except_qk_layernorm", fn=qwen3_no_wd_1d_cond
        )
        param_1d_key = ParamKey(predicate=param_1d_except_qk_ln)
        config_overrides[param_1d_key] = ParamGroupOverride(wd_mult=0.0)
    else:
        # Standard: skip weight decay for all 1D parameters
        param_length_1_match = ParamPredicate(
            name="param_len_1", fn=lambda param: len(param.shape) == 1
        )
        param_1d_key = ParamKey(predicate=param_length_1_match)
        config_overrides[param_1d_key] = ParamGroupOverride(wd_mult=0.0)

    return config_overrides if config_overrides else None


def setup_optimizer(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    model: Union[MegatronModule, list[MegatronModule]],
    use_gloo_process_groups: bool = False,
) -> tuple[MegatronOptimizer, OptimizerParamScheduler]:
    """Set up the optimizer and scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        model: The model to optimize
        use_gloo_process_groups: Whether to use Gloo process groups

    Returns:
        tuple containing the optimizer and scheduler
    """
    # For Qwen3-Next, mark q/k_layernorm params based on their names
    # This must be done before building config_overrides so the predicate can detect them
    if scheduler_config.no_weight_decay_cond_type == "qwen3_next":
        _mark_qk_layernorm_params(model)

    # Build config overrides for weight decay based on scheduler config
    config_overrides = _build_config_overrides(scheduler_config)

    if "muon" not in optimizer_config.optimizer and "soap" not in optimizer_config.optimizer:
        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )
    else:
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in optimizer_config.optimizer,
        )

    scheduler = _get_scheduler(optimizer_config, scheduler_config, optimizer)

    return optimizer, scheduler


def _get_scheduler(
    optimizer_config: OptimizerConfig, scheduler_config: SchedulerConfig, optimizer: MegatronOptimizer
) -> OptimizerParamScheduler:
    """Get the optimizer parameter scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        optimizer: The optimizer to schedule

    Returns:
        The optimizer parameter scheduler
    """
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=scheduler_config.lr_warmup_init,
        max_lr=optimizer_config.lr,
        min_lr=optimizer_config.min_lr,
        lr_warmup_steps=scheduler_config.lr_warmup_steps,
        lr_decay_steps=scheduler_config.lr_decay_steps,
        lr_decay_style=scheduler_config.lr_decay_style,
        start_wd=scheduler_config.start_weight_decay,
        end_wd=scheduler_config.end_weight_decay,
        wd_incr_steps=scheduler_config.wd_incr_steps,
        wd_incr_style=scheduler_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=scheduler_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=scheduler_config.override_opt_param_scheduler,
        wsd_decay_steps=scheduler_config.wsd_decay_steps,
        lr_wsd_decay_style=scheduler_config.lr_wsd_decay_style,
    )

    return scheduler
