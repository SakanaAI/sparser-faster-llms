import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch


@dataclass
class EnergyStats:
    avg_power_w: float
    energy_joules: float
    duration_s: float
    num_samples: int
    backend: str


class GPUPowerReader:
    """Lightweight helper that queries instantaneous GPU power draw."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = int(gpu_index)
        self.backend = None
        self._pynvml = None
        self._handle = None
        self.backend = self._init_backend()

    def _init_backend(self) -> Optional[str]:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._pynvml = pynvml
            return "pynvml"
        except Exception:
            pass

        if shutil.which("nvidia-smi") is not None:
            return "nvidia-smi"

        return None

    def read_power_w(self) -> Optional[float]:
        if self.backend == "pynvml":
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                return float(mw) / 1000.0
            except Exception:
                return None

        if self.backend == "nvidia-smi":
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu_index),
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                    timeout=1.0,
                )
                text = out.strip().splitlines()[0]
                return float(text)
            except Exception:
                return None

        return None


class GPUEnergyMonitor:
    """
    Background sampler that integrates power over time to estimate energy.
    Start it before the measured block and stop once the work completes.
    """

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.05):
        self.gpu_index = int(gpu_index)
        self.poll_interval_s = float(poll_interval_s)
        self.reader = GPUPowerReader(gpu_index=self.gpu_index)
        self.enabled = self.reader.backend is not None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Tuple[float, float]] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def _sample_once(self, ts: Optional[float] = None):
        ts = time.perf_counter() if ts is None else ts
        power = self.reader.read_power_w()
        if power is None:
            return
        self._samples.append((ts, power))

    def _poll_loop(self):
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.poll_interval_s)

    def start(self) -> bool:
        if not self.enabled or self._thread is not None:
            return False
        self._samples.clear()
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._sample_once(self._start_time)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._thread = None
        self._end_time = time.perf_counter()
        self._sample_once(self._end_time)

    def results(self) -> Optional[EnergyStats]:
        if not self.enabled or not self._samples:
            return None

        start_t = self._start_time or self._samples[0][0]
        end_t = self._end_time or self._samples[-1][0]
        duration = max(end_t - start_t, 0.0)
        if duration <= 0.0:
            return None

        energy = 0.0
        prev_t, prev_p = self._samples[0]
        for t, p in self._samples[1:]:
            dt = t - prev_t
            if dt > 0:
                energy += 0.5 * (p + prev_p) * dt
            prev_t, prev_p = t, p

        tail_dt = end_t - self._samples[-1][0]
        if tail_dt > 0:
            energy += self._samples[-1][1] * tail_dt

        avg_power = energy / duration if duration > 0 else float("nan")
        return EnergyStats(
            avg_power_w=avg_power,
            energy_joules=energy,
            duration_s=duration,
            num_samples=len(self._samples),
            backend=self.reader.backend or "unknown",
        )


def summarize_energy(stats: Optional[EnergyStats], reps: int, prefix: str = "") -> Dict[str, float]:
    if stats is None or reps <= 0:
        return {}

    energy_per_forward = stats.energy_joules / float(reps)
    return {
        f"{prefix}avg_power_w": float(stats.avg_power_w),
        f"{prefix}total_energy_j": float(stats.energy_joules),
        f"{prefix}energy_per_forward_j": float(energy_per_forward),
        f"{prefix}energy_duration_s": float(stats.duration_s),
        f"{prefix}energy_samples": int(stats.num_samples),
        f"{prefix}energy_backend": str(stats.backend),
    }


def merge_energy_stats(stats: List[EnergyStats]) -> Optional[EnergyStats]:
    if not stats:
        return
    total_energy = sum(s.energy_joules for s in stats)
    total_duration = sum(s.duration_s for s in stats)
    total_samples = sum(s.num_samples for s in stats)
    backend = stats[0].backend
    avg_power = (
        total_energy / total_duration if total_duration > 0 else float("nan")
    )
    return EnergyStats(
        avg_power_w=avg_power,
        energy_joules=total_energy,
        duration_s=total_duration,
        num_samples=total_samples,
        backend=backend,
    )


class MlpEnergyHooks:
    """
    Starts the monitor when the first MLP runs, stops after the last MLP
    in the forward pass. Keeps the monitor scoped to MLP compute.
    """

    def __init__(self, monitor: GPUEnergyMonitor, mlp_modules: List[torch.nn.Module],
                 sync_on_stop: bool = True):
        self.monitor = monitor
        self.sync_on_stop = sync_on_stop
        self.started = False
        self.handles = []
        if not mlp_modules or not monitor.enabled:
            return
        first = mlp_modules[0]
        last = mlp_modules[-1]
        self.handles.append(first.register_forward_pre_hook(self._pre_hook))
        self.handles.append(last.register_forward_hook(self._post_hook))

    def _pre_hook(self, module, _inputs):
        if self.started or not self.monitor.enabled:
            return
        self.monitor.start()
        self.started = True

    def _post_hook(self, module, _inputs, _output):
        if not self.monitor.enabled or not self.started:
            return
        if self.sync_on_stop and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.monitor.stop()
        self.started = False

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def plot_energy_bars(df, out_path, prefix: str = "", title: Optional[str] = None):
    power_col = f"{prefix}avg_power_w"
    energy_col = f"{prefix}energy_per_forward_j"
    if power_col not in df.columns or df[power_col].isna().all():
        return

    names = df["implementation"].to_list()
    x = list(range(len(names)))

    power_vals = df[power_col].to_list()
    energy_vals = (
        df[energy_col].to_list()
        if energy_col in df.columns
        else [math.nan] * len(names)
    )

    plt.figure()
    width = 0.38
    power_x = [xi - width / 2 for xi in x]
    energy_x = [xi + width / 2 for xi in x]

    plt.bar(power_x, power_vals, width=width, label="avg power (W)")
    if not all(math.isnan(v) for v in energy_vals):
        plt.bar(energy_x, energy_vals, width=width, label="energy per fwd (J)")

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("power (W) / energy (J)")
    plt.title(title or "GPU power and energy per forward")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()
