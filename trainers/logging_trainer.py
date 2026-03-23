"""Trainer wrapper that logs metrics exposed by sparse models."""

import inspect
import os
import time
from typing import Callable, Optional

import numpy as np
import torch
from transformers import Trainer


def is_tensor(value) -> bool:
    return isinstance(value, torch.Tensor)


class TrainerWithModelMetrics(Trainer):
    def __init__(
        self,
        *args,
        init_callback_functions: Optional[list[str | Callable[..., None]]] = None,
        measure_memory: bool = False,
        measure_time_per_step: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.measure_memory = measure_memory
        self.measure_time_per_step = measure_time_per_step
        self._last_log_time = (
            time.perf_counter() if self.measure_time_per_step else None
        )
        self._last_log_step = (
            self.state.global_step if self.measure_time_per_step else None
        )
        if self.measure_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.initialize_callbacks(init_callback_functions)

    def initialize_callbacks(
        self,
        init_callback_functions: Optional[list[str | Callable[..., None]]],
    ) -> None:
        self.logging_model = hasattr(self.model, "get_and_flush_metrics")

        if hasattr(self.model, "save_trainer_reference"):
            self.model.save_trainer_reference(self)

        if not init_callback_functions:
            return

        for fn in init_callback_functions:
            if isinstance(fn, str):
                if not hasattr(self.model, fn):
                    raise AttributeError(f"model has no attribute '{fn}'")
                fn = getattr(self.model, fn)
            if not callable(fn):
                raise TypeError(f"callback {fn!r} is not callable")
            if "trainer" not in inspect.signature(fn).parameters:
                raise TypeError(
                    f"callback {fn.__name__} must accept a 'trainer' keyword"
                )
            fn(trainer=self)

    def get_model_custom_metric(self) -> dict[str, float]:
        model_dict = self.model.get_and_flush_metrics()
        ga = getattr(self.args, "gradient_accumulation_steps", 1) or 1
        ws = getattr(self.args, "world_size", None)
        if ws is None:
            ws = int(os.environ.get("WORLD_SIZE", "1"))
        multiplier = ga * ws

        logged_dict = {}
        for log_name, log_value in model_dict.items():
            if is_tensor(log_value):
                log_value = log_value.mean().item()
            elif isinstance(log_value, (list, tuple)):
                log_value = np.mean(log_value)
            else:
                log_value = float(log_value)
            logged_dict[log_name] = log_value
            logged_dict[f"{log_name}/unscaled"] = log_value * multiplier
        return logged_dict

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if self.measure_memory and torch.cuda.is_available():
            device = getattr(self.args, "device", None)
            if isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.synchronize(device)
                logs["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)

        if self.measure_time_per_step:
            now = time.perf_counter()
            if self._last_log_time is not None and self._last_log_step is not None:
                step_diff = max(1, self.state.global_step - self._last_log_step)
                logs["time_per_step"] = (now - self._last_log_time) / step_diff
            self._last_log_time = now
            self._last_log_step = self.state.global_step

        if self.logging_model:
            logs.update(self.get_model_custom_metric())
        super().log(logs, start_time=start_time)
