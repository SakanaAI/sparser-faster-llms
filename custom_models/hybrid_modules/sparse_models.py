import abc
import os
from typing import List, Mapping, Optional, Dict, Tuple, Callable

import sys
import deepspeed
import torch
import torch.nn as nn
import torch.distributed as dist

from sparse_ops_loader import load_sparse_ops
load_sparse_ops()

# modify this value to dynamically switch between sparse/dense ops
L0_CUTOFF = 70

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Config,
)

import logging

logger = logging.getLogger(__name__)

def get_total_devices():
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1


def get_tensor_stats_for_logging(
    prefix: str, t: torch.Tensor | List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    if not isinstance(t, torch.Tensor) and isinstance(t, list):
        t = torch.stack([ti.detach() for ti in t], dim=0)
    t_flat = t.detach().view(-1)
    return {
        f"{prefix}/min": t_flat.min(),
        f"{prefix}/max": t_flat.max(),
        f"{prefix}/mean": t_flat.mean(),
        f"{prefix}/std": t_flat.std(),
    }


class NullTracker:
    '''Fast no-op tracker for compatibility'''
    def __init__(
        self,
        train_batch_size: int,
        max_seq_len: int,
        tokens_until_dead: Optional[int] = None,
    ) -> None:
        pass

    def register_layer(
        self, layer_idx: int,
        width: int,
        device: torch.device,
    ) -> None:
        pass

    def start_forward_pass(self) -> None:
        pass

    def record_activations(
        self,
        layer_idx: int,
        **kwargs,
    ) -> None:
        pass

    def finalize_forward_pass(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return torch.tensor(0.0), {}


class SparsityTracker:
    '''Computes L1 auxiliary loss, which tokens are dead or alive and l0 and
       other statistics about neuron activation usage not used in training.'''

    def __init__(
        self,
        train_batch_size: int,
        max_seq_len: int,
        per_device_train_batch_size: int,
        # number of tokens until a neuron is considered dead if it has
        # not activated, if None, defaults to # of tokens in a batch
        tokens_until_dead: Optional[int] = None,
        gated: bool = False,
    ) -> None:
        self._last_activated: Dict[int, torch.Tensor] = {}
        self._l1_per_layer: List[torch.Tensor] = []
        self._l0_per_layer: List[torch.Tensor] = []

        self._batch_tokens = int(
            (train_batch_size * max_seq_len) // get_total_devices())
        
        self._microbatch_tokens = int(
            (per_device_train_batch_size * max_seq_len)
        )

        print(
            f"SparsityTracker: batch tokens {self._batch_tokens}, "
            f"microbatch tokens {self._microbatch_tokens}, "
            f"ratio {self._batch_tokens / self._microbatch_tokens}"
        )

        if tokens_until_dead is None:
            tokens_until_dead = self._batch_tokens
        #     if microbatches_until_dead is not None:
        #         tokens_until_dead *= int(microbatches_until_dead)
        # else:
        #     assert microbatches_until_dead is None, (
        #         "Cannot specify both tokens_until_dead and "
        #         "microbatches_until_dead"
        #     )
        self._tokens_until_dead = int(tokens_until_dead)
        self._number_of_layers: Optional[int] = None
        self._gated = gated
        self.active = False

    def register_layer(
        self,
        layer_idx: int,
        width: int,
        device: torch.device,
    ) -> None:
        if layer_idx in self._last_activated:
            return
        self._last_activated[layer_idx] = torch.zeros(
            int(width),
            dtype=torch.int64,
            device=device,
        )

    def start_forward_pass(self) -> None:
        # clear per-forward buffers and activate tracking
        self.active = True
        self._l1_per_layer.clear()
        self._l0_per_layer.clear()

    def record_activations(
        self,
        layer_idx: int,
        absolute_activations: torch.Tensor,
    ) -> None:

        # if we're in checkpoint backward recomputation, don't touch state
        if (not self.active) and torch.is_grad_enabled():
            return

        if layer_idx not in self._last_activated:
            self.register_layer(
                layer_idx=layer_idx,
                width=absolute_activations.shape[-1],
                device=absolute_activations.device,
            )


        # to use for l1 loss since always >= 0 with ReLU
        l1_average = absolute_activations.sum(dim=-1).mean()
        self._l1_per_layer.append(l1_average)
        if not self.active:
            return

        # fraction of tokens where each neuron is active in this microbatch
        is_active = absolute_activations.detach() > 0
        l0_per_neuron = is_active.float().mean(dim=(0, 1))
        l0_average = l0_per_neuron.sum(-1)
        #print("l0 is", l0_average)
        self._l0_per_layer.append(l0_average)

        last_activated = torch.where(
            # False,
            # l0_per_neuron > 1e6,
            l0_per_neuron > 0,
            # expected tokens since last activation
            # expected_trailing_gap.to(
            #         self._last_activated[layer_idx].dtype),
            torch.zeros_like(self._last_activated[layer_idx]),
            # increment by number of tokens in microbatch
            self._last_activated[layer_idx] + self._microbatch_tokens,
        )

        self._last_activated[layer_idx] = last_activated

    def record_l0_l1(self, layer_idx, l0, l1):
        # if we're in checkpoint backward recomputation, don't touch state
        if (not self.active) and torch.is_grad_enabled():
            return

        if layer_idx not in self._last_activated:
            self.register_layer(
                layer_idx=layer_idx,
                width=5632, # TODO HACK
                device=l0.device,
            )


        # to use for l1 loss since always >= 0 with ReLU
        l1_average = l1
        self._l1_per_layer.append(l1_average)
        if not self.active:
            return

        # fraction of tokens where each neuron is active in this microbatch
        assert layer_idx == len(self._l0_per_layer)
        l0_average = l0
        self._l0_per_layer.append(l0_average)

    def get_alive_neurons(self) -> List[torch.Tensor]:
        if self._number_of_layers is None:
            raise ValueError(
                "Must call finalize_forward_pass before get_alive_neurons"
            )
        alive_list: List[torch.Tensor] = []
        for i in range(self._number_of_layers):
            last_activated = self._last_activated[i]
            alive_mask = (
                last_activated < self._tokens_until_dead
            ).to(torch.float32)
            alive_list.append(alive_mask)
        return alive_list

    def finalize_forward_pass(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats_to_log: Dict[str, torch.Tensor] = {}
        if self._number_of_layers is None:
            self._number_of_layers = len(self._last_activated)

        if not self._l1_per_layer:
            l1_auxiliary_loss = 0.0
            return l1_auxiliary_loss, stats_to_log

        l1_auxiliary_loss = torch.mean(torch.stack(self._l1_per_layer))

        if not self.active:
            return l1_auxiliary_loss, stats_to_log
        self.active = False
        l0_stats = get_tensor_stats_for_logging(
            "activation_l0", self._l0_per_layer
        )
        l1_stats = get_tensor_stats_for_logging(
            "activation_l1", self._l1_per_layer
        )

        alive_list = self.get_alive_neurons()
        alive_fractions = [alive_mask.mean() for alive_mask in alive_list]
        alive_stats = get_tensor_stats_for_logging(
            "activation_alive", alive_fractions
        )

        stats_to_log.update(l0_stats)
        stats_to_log.update(l1_stats)
        stats_to_log.update(alive_stats)
        return l1_auxiliary_loss, stats_to_log

activations = {}

class SparseMLP(nn.Module):
    """FFN supporting multiple variants."""

    def __init__(
        self,
        layer_idx: int,
        config: PretrainedConfig,
        gated: bool,
        tracker: SparsityTracker,
        new_intermediate_size: Optional[int] = None,
        compile_mlp: bool = False,
        execution_logic: str | Callable = "training",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.gated = gated
        self.tracker = tracker
        self.hidden_size = config.hidden_size
        if new_intermediate_size is None:
            self.intermediate_size = config.intermediate_size
        else:
            self.intermediate_size = new_intermediate_size

        mlp_bias = getattr(config, "mlp_bias", False)
        assert not mlp_bias

        if gated:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=mlp_bias
            )
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=mlp_bias
            )
            self.down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=mlp_bias
            )
        else:
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=mlp_bias
            )
            self.down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=mlp_bias
            )
        self.act_fn = nn.ReLU()

        self._training_mode = False
        if isinstance(execution_logic, str):
            if execution_logic == "inference":
                self._forward_inference = self._forward_mlp_inference_logic
            elif execution_logic == "training":
                self._forward_inference = None
                self._training_mode = True
            else:
                raise ValueError(
                    f"Unknown execution_logic {execution_logic}, "
                    "if string, must be 'inference' or 'training'"
                )
        elif isinstance(execution_logic, Callable):
            self._forward_inference = execution_logic
        else:
            raise ValueError(
                "execution_logic must be string or Callable"
            )
        if compile_mlp:
            self._forward_inference = torch.compile(self._forward_inference)

        self._use_fast = False

        # Sparse kernels expect down_proj.weight in [intermediate, hidden]
        # layout, opposite of nn.Linear's default. Bake the transpose in at
        # construction so from-scratch models start in the right layout; HF
        # checkpoints (loaded via hydra_utils.load_model with from_pretrained=True)
        # are transposed at load time instead.
        self.down_proj.weight.data = self.down_proj.weight.data.t().contiguous()

    def _forward_mlp_inference_logic(self, hidden_states: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.gated:
            gate_projections = self.gate_proj(hidden_states)
            up_projections = self.up_proj(hidden_states)
            hidden_states = self.act_fn(gate_projections) * up_projections
        else:
            hidden_states = self.act_fn(self.up_proj(hidden_states))
        return self.down_proj(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._training_mode:
            if self._use_fast:
                outs, l0, l1, _, _, _ = torch.ops.sparse_ops.ff_forward_gated.default(
                    hidden_states, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)
                self.tracker.record_l0_l1(self.layer_idx, l0, l1)
                return outs
            if self.gated:
                gate_activation = torch.einsum("bsd,hd->bsh", hidden_states, self.gate_proj.weight)
                up_activation = torch.einsum("bsd,hd->bsh", hidden_states, self.up_proj.weight)
                activation = self.act_fn(gate_activation) * up_activation
                absolute_activation = torch.abs(activation)
            else:
                activation = self.act_fn(self.up_proj(hidden_states))
                absolute_activation = activation

            self.tracker.record_activations(
                layer_idx=self.layer_idx,
                absolute_activations=absolute_activation,
            )
            return torch.einsum("bsh,hd->bsd", activation, self.down_proj.weight)
        else:
            return self._forward_inference(hidden_states)

    def enable_fast(self, use_fast):
        self._use_fast = use_fast

    def load_state_dict(self, state):
        print(state.keys())
        raise RuntimeError("state")

class MetricsAccumulator:
    def __init__(self):
        self.metrics: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

    def add_metrics(self, metrics: Mapping[str, torch.Tensor]):
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = v.detach()
                self.counts[k] = 1
            else:
                self.metrics[k] += v.detach()
                self.counts[k] += 1

    def get_and_flush_metrics(self) -> Mapping[str, float]:
        out: Dict[str, float] = {}
        for k, v in self.metrics.items():
            out[k] = (v / self.counts[k]).item()
        self.metrics = {}
        self.counts = {}
        return out


class Scheduler:
    def __init__(
        self,
        final_value: Optional[float],
        post_warmup_value: float,
        start_warmup_step: int,
        start_decay_step: int,
        end_decay_step: Optional[int],
    ) -> None:
        if start_warmup_step < 0:
            raise ValueError("start_warmup_step must be >= 0")
        if start_decay_step < 0:
            raise ValueError("start_decay_step must be >= 0")
        if start_warmup_step > start_decay_step:
            raise ValueError("expected start_warmup_step <= start_decay_step")

        if end_decay_step is not None:
            if end_decay_step < 0:
                raise ValueError("end_decay_step must be >= 0")
            if start_decay_step > end_decay_step:
                raise ValueError("expected start_decay_step <= end_decay_step")

        self.final_value = None if final_value is None else float(final_value)
        self.post_warmup_value = float(post_warmup_value)
        self.start_warmup_step = int(start_warmup_step)
        self.warmup_length = int(start_decay_step - start_warmup_step)
        self.start_decay_step = int(start_decay_step)
        self.end_decay_step = None if end_decay_step is None else int(end_decay_step)

    def __call__(self, step: int) -> float:
        step_i = int(step)

        if step_i <= self.start_warmup_step:
            return 0.0
        elif step_i < self.start_decay_step:
            step_difference = step_i - self.start_warmup_step
            warmup_frac = step_difference / self.warmup_length
            return float(warmup_frac * self.post_warmup_value)

        if self.final_value is None or self.end_decay_step is None:
            return self.post_warmup_value

        if step_i >= self.end_decay_step:
            return self.final_value

        decay_span = self.end_decay_step - self.start_decay_step
        if decay_span <= 0:
            return self.final_value

        decay_frac = (step_i - self.start_decay_step) / decay_span
        return float(
            self.post_warmup_value
            + (self.final_value - self.post_warmup_value) * decay_frac
        )

class SparseModelForCausalLM(abc.ABC):
    """Model with sparse ReLU MLPs"""

    def __init__(
        self,
        config: PretrainedConfig,
        gated: bool,
        tracker: SparsityTracker,
        new_intermediate_size: Optional[int] = None,
        l1_coeff: float = 3e-4,
        l1_start_warmup_step: Optional[int] = None,
        l1_end_warmup_step: Optional[int] = None,
        reinitialize: bool = False,
        mlp_execution_logic: str = 'training',
        baseline_run: bool = False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.trainer = None
        self.config = config
        self.gated = gated
        self.tracker = tracker
        self.new_intermediate_size = new_intermediate_size
        self.l1_coeff = l1_coeff
        if l1_start_warmup_step is not None:
            assert l1_end_warmup_step is not None
            self.l1_scheduler = Scheduler(
                final_value=None,
                start_warmup_step=l1_start_warmup_step,
                start_decay_step=l1_end_warmup_step,
                end_decay_step=None,
                post_warmup_value=l1_coeff,
            )
        else:
            assert l1_end_warmup_step is None
            self.l1_scheduler = None
        self.metrics_accumulator = MetricsAccumulator()
        self.reinitialize = reinitialize
        self.mlp_execution_logic = mlp_execution_logic
        self.baseline_run = baseline_run
        if self.baseline_run:
            assert self.l1_coeff == 0.0, (
                "Baseline run specified, l1_coeff must be 0.0"
            )
        self._iter = 0

    def get_and_flush_metrics(self) -> dict:
        stored_metrics = self.metrics_accumulator.get_and_flush_metrics()
        return stored_metrics

    def save_trainer_reference(self, trainer):
        self.trainer = trainer

    def replace_mlp_modules(
            self,
            custom_class: Optional[type] = None,
            reinitialize: Optional[bool] = None,
            ):
        if reinitialize is None:
            reinitialize = self.reinitialize
        if reinitialize:
            assert hasattr(self.config, "initializer_range"), (
                "Config must have initializer_range attribute to "
                "reinitialize weights."
            )
            if custom_class is not None:
                logger.warning(
                    "Reinitialization requested but custom_class is "
                    "provided, skipping reinitialization."
                )

            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range
                    )
                    if module.bias is not None:
                        module.bias.data.zero_()
        else:
            def _init_weights(module):
                pass
        for i, layer in enumerate(self.model.layers):
            original_mlp = layer.mlp
            if custom_class is not None:
                sparse_mlp = custom_class(
                    gate_linear=original_mlp.gate_proj,
                    up_linear=original_mlp.up_proj,
                    down_linear=original_mlp.down_proj,
                    layer_idx=i,
                    config=self.config,
                )
            else:
                sparse_mlp = SparseMLP(
                    layer_idx=i,
                    config=self.config,
                    gated=self.gated,
                    tracker=self.tracker,
                    new_intermediate_size=self.new_intermediate_size,
                    execution_logic=self.mlp_execution_logic,
                )
                sparse_mlp.apply(_init_weights)
            self.model.layers[i].mlp = sparse_mlp
            del original_mlp
            torch.ops.sparse_ops_config.set_discard_overflow(2)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        collect_stats: bool = False,
        **kwargs,
    ):
        if (labels is not None or collect_stats) and not self.baseline_run:
            assert self.mlp_execution_logic == "training", (
                "Can only collect sparsity statistics and L1 loss during when "
                "the execution logic is set to 'training'."
            )
            self.tracker.start_forward_pass()

        torch.cuda.nvtx.range_push("forward")
        output: CausalLMOutputWithPast = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        if self.baseline_run:
            return output

        if labels is not None or collect_stats:
            l1_auxiliary_loss, sparsity_stats = (
                self.tracker.finalize_forward_pass()
            )
            self.metrics_accumulator.add_metrics(sparsity_stats)
        else:
            l1_auxiliary_loss = torch.tensor(0.0, device=output.loss.device)
        if labels is not None:
            if self.l1_scheduler is not None:
                step = int(self.trainer.state.global_step)
                l1_coeff = self.l1_scheduler(step)
            else:
                l1_coeff = self.l1_coeff

            self.metrics_accumulator.add_metrics(
                {
                    "loss/ce_loss": output.loss.detach(),
                    "loss/l1_loss": l1_auxiliary_loss.detach(),
                    "loss/l1_coeff": torch.tensor(l1_coeff),
                }
            )
            output.loss = output.loss + l1_coeff * l1_auxiliary_loss

        torch.cuda.nvtx.range_pop()
        self._iter +=1
        if self._iter % 20 == 0:
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer.mlp.tracker, "_l0_per_layer") and len(layer.mlp.tracker._l0_per_layer) > 0:
                    layer_l0 = layer.mlp.tracker._l0_per_layer[i]
                    layer.mlp.enable_fast(layer_l0 < L0_CUTOFF)
        return output


def _create_sparse_config_class(
    BaseConfigClass: type[PretrainedConfig],
) -> type[PretrainedConfig]:
    class SparseConfig(BaseConfigClass):
        model_type = f"{BaseConfigClass.model_type}_sparse_relu"

        def __init__(
            self,
            sparsity_train_batch_size: int = 1,
            sparsity_per_device_train_batch_size: int = 1,
            sparsity_max_seq_len: int = 2048,
            sparsity_tokens_until_dead: Optional[int] = None,
            # sparsity_microbatches_until_dead: Optional[int] = None,
            sparsity_l1_coeff: float = 3e-4,
            sparsity_start_warmup_step: Optional[int] = None,
            sparsity_end_warmup_step: Optional[int] = None,
            sparsity_new_intermediate_size: Optional[int] = None,
            sparsity_gated_mlp: bool = True,
            mlp_bias: bool = False,
            sparsity_reinitialize: bool = False,
            sparsity_do_track: bool = True,
            sparsity_mlp_execution_logic: str = 'training',
            sparsity_baseline_run: bool = False,
            **kwargs,
        ):
            self.sparsity_train_batch_size = sparsity_train_batch_size
            self.sparsity_per_device_train_batch_size = (
                sparsity_per_device_train_batch_size
            )   
            self.sparsity_max_seq_len = sparsity_max_seq_len
            self.sparsity_tokens_until_dead = sparsity_tokens_until_dead
            # self.sparsity_microbatches_until_dead = (
            #     sparsity_microbatches_until_dead
            # )
            self.sparsity_l1_coeff = sparsity_l1_coeff
            self.sparsity_start_warmup_step = sparsity_start_warmup_step
            self.sparsity_end_warmup_step = sparsity_end_warmup_step
            self.sparsity_new_intermediate_size = (
                sparsity_new_intermediate_size
            )
            self.sparsity_gated_mlp = sparsity_gated_mlp
            self.sparsity_reinitialize = sparsity_reinitialize
            self.sparsity_do_track = sparsity_do_track
            self.sparsity_mlp_execution_logic = sparsity_mlp_execution_logic
            self.sparsity_baseline_run = sparsity_baseline_run
            if not hasattr(self, "mlp_bias"):
                self.mlp_bias = mlp_bias
            super().__init__(**kwargs)

        @classmethod
        def from_base_config(cls, base_config: BaseConfigClass):
            return cls(**base_config.to_dict())

    SparseConfig.__name__ = f"Sparse{BaseConfigClass.__name__}"
    AutoConfig.register(SparseConfig.model_type, SparseConfig)
    SparseConfig.register_for_auto_class()
    return SparseConfig


def _create_sparse_model_class(
    BaseModelClass: type[PreTrainedModel],
    BaseConfigClass: type[PretrainedConfig],
) -> Tuple[type[PreTrainedModel], type[PretrainedConfig]]:
    SparseConfigClass = _create_sparse_config_class(BaseConfigClass)

    class SparseModel(SparseModelForCausalLM, BaseModelClass):
        config_class = SparseConfigClass

        def __init__(
            self,
            config: PretrainedConfig,
            *args,
            **kwargs,
        ):
            if config.sparsity_do_track:
                tracker_class = SparsityTracker
            else:
                tracker_class = NullTracker
                assert config.sparsity_l1_coeff == 0.0, (
                    "sparsity_do_track is False, is only compatible with "
                    "baseline training where sparsity_l1_coeff must be 0.0"
                )

            tracker = tracker_class(
                train_batch_size=config.sparsity_train_batch_size,
                per_device_train_batch_size=(
                    config.sparsity_per_device_train_batch_size),
                max_seq_len=config.sparsity_max_seq_len,
                tokens_until_dead=config.sparsity_tokens_until_dead,
                # microbatches_until_dead=(
                #     config.sparsity_microbatches_until_dead
                # ),
            )
            super().__init__(
                config=config,
                gated=config.sparsity_gated_mlp,
                tracker=tracker,
                new_intermediate_size=(
                    config.sparsity_new_intermediate_size
                ),
                l1_coeff=config.sparsity_l1_coeff,
                reinitialize=config.sparsity_reinitialize,
                l1_start_warmup_step=config.sparsity_start_warmup_step,
                l1_end_warmup_step=config.sparsity_end_warmup_step,
                mlp_execution_logic=config.sparsity_mlp_execution_logic,
                baseline_run=config.sparsity_baseline_run,
                *args,
                **kwargs,
            )
            self.replace_mlp_modules()

    architecture_name = f"Sparse{BaseModelClass.__name__}"
    SparseModel.__name__ = architecture_name

    AutoModelForCausalLM.register(SparseModel.config_class, SparseModel)
    SparseModel.register_for_auto_class("AutoModelForCausalLM")

    return SparseModel, SparseConfigClass


SparseLlamaForCausalLM, SparseLlamaConfig = _create_sparse_model_class(
    LlamaForCausalLM, LlamaConfig
)

SparseQwen2ForCausalLM, SparseQwen2Config = _create_sparse_model_class(
    Qwen2ForCausalLM, Qwen2Config
)
