# flux2-tpu

JAX-native inference for [FLUX.2 Klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
targeting free-tier TPU (Colab v5e-1, Kaggle v5e-8). No PyTorch
dependency anywhere in this codebase.

Weights live separately at
[johaness14/flux2-klein-4b-jax](https://huggingface.co/johaness14/flux2-klein-4b-jax),
converted from the original PyTorch checkpoints by a one-time
conversion pipeline (not part of this repository).

## No safety filtering

This codebase includes no content moderation, NSFW classifier, or
prompt/output filtering of any kind. Anyone using it is solely
responsible for how they use it.

## Status

Under active development, built incrementally with tests at each step
rather than as one large unverified drop.

| Component | Status |
|---|---|
| Configuration, checkpoint download/restore | Done |
| VAE decoder | Not started |
| Text encoder (Qwen3-4B) | Not started |
| Diffusion transformer | Not started |
| Sampling loop | Not started |
| Notebook runner | Not started |

There is no working `generate()` yet. This repository is not usable
for image generation at this point in its development.

## Layout

```
flux2_klein/
├── logging_setup.py    # dual console + file logging
├── config.py            # residency strategy, resolution buckets, checkpoint source
└── checkpoint.py         # download and restore the JAX-native bundle
tests/
├── test_config.py
└── test_checkpoint.py
```

## Setup

```bash
pip install -r requirements.txt
```

## License

Apache-2.0.
