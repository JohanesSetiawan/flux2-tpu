# flux2-tpu

JAX-native inference for [FLUX.2 Klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
targeting free-tier TPU: Google Colab v5e-1 and Kaggle v5e-8. No
PyTorch dependency anywhere in this codebase.

Weights live separately at
[johaness14/flux2-klein-4b-jax](https://huggingface.co/johaness14/flux2-klein-4b-jax),
converted from the original PyTorch checkpoints by a one-time
conversion pipeline that is not part of this repository.

## No safety filtering

This codebase includes no content moderation, NSFW classifier, or
prompt/output filtering of any kind. The reference FLUX.2 pipeline's
moderation layer was intentionally not ported. Anyone using this
codebase is solely responsible for how they use it.

## Status

Under active development, built incrementally with tests at each step
rather than as one large unverified drop. **There is no working
`generate()` yet; this repository cannot produce images at this point
in its development.**

| Component | Status |
|---|---|
| Configuration, residency strategy, resolution buckets | Done |
| Checkpoint download and restore | Done |
| Logging | Done |
| VAE layer primitives | Done |
| VAE residual block, attention block, full decoder | Not started |
| Text encoder (Qwen3-4B) | Not started |
| Diffusion transformer | Not started |
| Sampling loop and pipeline | Not started |
| Notebook runner, ipywidgets and Gradio interfaces | Not started |

## Design summary

**Model.** FLUX.2 Klein-4B is a 4-step distilled rectified-flow
text-to-image model. The number of sampling steps and the guidance
value are fixed by distillation and are not tunable; the reference
implementation rejects attempts to change them. Speed comes from
compilation and scheduling, never from altering the sampling
mathematics.

**Precision.** Nothing is quantized. The transformer runs in bf16,
which is the checkpoint's own dtype, and the VAE decoder runs in fp32,
matching the reference decode path. Output is intended to be identical
to the reference implementation.

**Memory.** Fitting a 16 GB single-chip accelerator is handled by
choosing where each component lives rather than by reducing precision:

- `FULLY_RESIDENT`: all three components stay in accelerator memory.
  Suitable for Kaggle v5e-8 (128 GB aggregate).
- `SWAPPED`: the transformer and VAE decoder stay resident while the
  text encoder lives in host RAM and enters accelerator memory only
  while encoding a prompt. Suitable for Colab v5e-1 at full precision.

The strategy defaults to automatic selection based on the number of
visible JAX devices, and can be overridden explicitly.

**Resolutions.** Three buckets are supported, all sitting below the
4300-image-token threshold at which the reference sampling schedule
switches to a formula this checkpoint's 4-step distillation was not
tuned against:

| Resolution | Aspect ratio | Image tokens |
|---|---|---|
| 1024 x 1024 | 1:1 | 4096 |
| 1360 x 768 | 16:9 | 4080 |
| 768 x 1360 | 9:16 | 4080 |

## Layout

```
flux2_klein/
├── logging_setup.py   # dual console and file logging
├── config.py          # every tunable value, as dataclasses
├── checkpoint.py      # download and restore the JAX-native bundle
└── layers.py          # VAE layer primitives (pure functions)
tests/
├── run_all_tests.py   # single entry point, writes a full log
├── test_config.py
├── test_checkpoint.py
└── test_layers.py
```

Numeric code is pure: it takes arrays and a configuration object,
returns arrays, and performs no IO or logging. Orchestration code
performs IO and logs every stage to both the console and a text file.
Interfaces (notebook, widgets) will contain no logic and only call into
the package.

## Setup

```bash
pip install -r requirements.txt
```

## Running the tests

```bash
python -m tests.run_all_tests
python -m tests.run_all_tests --log-file path/to/log.txt
```

Regression tests compare against float64 reference oracles and
therefore need 64-bit mode enabled:

```bash
JAX_ENABLE_X64=1 python -m tests.run_all_tests
```

Tests come in two tiers. Smoke tests check shape, dtype and basic
execution. Regression tests check numerical correctness against
independently implemented oracles, swept across a matrix of shapes
generated at run time rather than compared against stored golden
arrays.

## Contributing

Read [AGENTS.md](AGENTS.md) first. It documents the architectural
conventions, the verified model facts this implementation depends on,
and the hazards that have already caused bugs here.

## License

Apache-2.0.
