<h1 align="center">
  <a href="https://github.com/SakanaAI/sparser-faster-llms">
<img src="figures/logo.png" width="300" /></a><br>
<b>Sparser, Faster, Lighter Transformer Language Models</b><br>
</h1>
<p align="center">
  📚 <a href="https://arxiv.org/abs/coming-soon">[Paper]</a> |
  🤗 <a href="https://huggingface.co/collections/SakanaAI/sparser-faster-lighter-transformers">[Checkpoints]</a>
  <!-- 🐠 <a href="https://sakana.ai/blog/rlt">[Blog (coming soon)]</a> -->
</p>

This repository contains the reference code for the paper **Sparser, Faster, Lighter Transformer Language Models**. It includes sparse training code and our custom CUDA kernels designed for H100 GPUs for sparse models, leveraging the TwELL packing format.

## Installation

The repository expects a CUDA 12.8+ environment:

```bash
git clone https://github.com/SakanaAI/sparser-faster-llms.git
cd sparser-faster-llms
bash scripts/install.sh
# or uv
# python -m venv .venv
# source .venv/bin/activate
# bash scripts/install.sh --uv
```

## Repo Structure

```text
.
├── accelerate_configs/          # Accelerate + DeepSpeed launch configs
├── benchmark_inference.py       # Minimal torch vs TwELL inference benchmark
├── benchmark_base.py            # Small benchmark helpers
├── cfgs/                        # Hydra configs for model/data/training
├── custom_data/                 # Pretraining dataset utilities
├── custom_models/
│   ├── sparse_models.py         # Sparse model definitions
│   ├── sparse_testing_utils.py  # Sparse -> HF / TwELL conversion helpers
│   └── twell_modules/           # TwELL CUDA kernels
├── energy_utils.py              # Optional GPU energy measurement helpers
├── launch.sh                    # Main multi-GPU training entrypoint
├── load_dataset.py              # Dataset loading glue
├── scripts/install.sh           # Minimal installation script
├── train.py                     # Hydra training entrypoint
└── trainers/
    └── logging_trainer.py       # Trainer used by the public training path
```

## Roadmap

- [x] Sparse model training code
- [x] TwELL inference kernels
- [ ] Efficient TwELL training kernels

## Inference Benchmarking

We release pretrained sparse checkpoints on the Hugging Face Hub at:

- `SakanaAI/SparseLM0.5B`
- `SakanaAI/SparseLM1B`
- `SakanaAI/SparseLM1.5B`
- `SakanaAI/SparseLM2B`

You can benchmark our kernels against the Hugging Face PyTorch reference with our benchmarking scripts `benchmark_inference.py`, e.g.: 

```bash
python benchmark_inference.py \
  --model-path SakanaAI/SparseLM1.5B \
  --reps 500 \
  --warmup-reps 5 \
  --out-csv results/benchmark_inference/SparseLM1.5B.csv
```

To also measure GPU energy during the benchmark loop:

```bash
python benchmark_inference.py \
  --model-path SakanaAI/SparseLM1.5B \
  --reps 500 \
  --warmup-reps 5 \
  --measure-energy \
  --out-csv results/benchmark_inference/SparseLM1.5B_energy.csv
```

You can benchmark your own local sparse models by overriding `--model-path /path/to/local/checkpoint_dir`.

## Training (Torch)

We provide simple functionality using standard PyTorch for Sparse training:

```bash
./launch.sh <num_gpus> <run_cfg> [zero1|offload|offload_optim] [hydra overrides...]
```

We provide premade Hydra configs to obtain sparse models at different sizes:

- `sparsity_gated_0p5b`
- `sparsity_gated_1b`
- `sparsity_gated_1p5b`
- `sparsity_gated_2b`

For training on H100 GPUs, we recommend the default settings, `zero1` optimization, and no parameter offloading:

```bash
./launch.sh 8 sparsity_gated_1p5b zero1
```

The default logging functionality saves results both locally and to [Weights & Biases](https://wandb.ai/). To disable Weights & Biases logging, please modify the provided configuration files with:

```yaml
report_to: null
```

## Citation

If you find our work or this repository useful and want to cite our paper, you can use the following:

```bibtex
@article{sakana2026sparser,
}
```
