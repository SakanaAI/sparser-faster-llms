#!/usr/bin/env python3

"""Load and inspect the configured pretraining dataset."""

import argparse
import os

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf


def get_total_devices() -> int:
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1


def compute_accumulation_steps(
    train_batch_size: int,
    per_device_train_batch_size: int,
) -> int:
    total_devices = get_total_devices()
    divisor = per_device_train_batch_size * total_devices
    steps = train_batch_size / divisor
    if not steps.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"per_device_batch*total_devices={divisor}"
        )
    return int(steps)


def build_cfg(run_cfg: str, overrides: list[str]):
    with initialize(version_base=None, config_path="cfgs", job_name="load_dataset"):
        cfg = compose(
            config_name="train",
            overrides=[f"run_cfg@_global_={run_cfg}", *overrides],
        )
    if OmegaConf.is_missing(cfg, "gradient_accumulation_steps"):
        cfg.gradient_accumulation_steps = compute_accumulation_steps(
            train_batch_size=cfg.train_batch_size,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
        )
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Instantiate the tokenizer and pretraining dataset for a run config."
    )
    parser.add_argument(
        "--run-cfg",
        type=str,
        default="sparsity_gated_0p5b",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides.",
    )
    args = parser.parse_args()

    cfg = build_cfg(args.run_cfg, args.overrides)
    tokenizer = hydra.utils.instantiate(cfg.make_tokenizer_fn)
    datasets = hydra.utils.instantiate(cfg.make_dataset_fn, tokenizer=tokenizer)

    print(OmegaConf.to_yaml(cfg))
    print("train_dataset:", type(datasets.get("train_dataset")))
    print("eval_dataset:", type(datasets.get("eval_dataset")))


if __name__ == "__main__":
    main()
