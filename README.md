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

## Quick start

Open `notebooks/generate.ipynb` in Kaggle or Colab with a TPU
accelerator selected, and run it top to bottom. It clones this
repository, downloads the weights, warms up the compiler, and offers
three ways to generate: in-notebook controls, a browser interface, or a
direct call.

## Status

Every component is implemented and verified against the reference. What
has **not** been established is performance: none of it has run on real
TPU hardware, so every latency and memory figure in this project is
arithmetic rather than measurement.

| Component | Status |
|---|---|
| Configuration, residency strategy, resolution buckets | Done |
| Checkpoint download and restore | Done |
| Logging | Done |
| VAE layer primitives | Done |
| VAE residual and attention blocks | Done |
| VAE full decoder assembly | Done |
| VAE parity against the reference implementation | Done, 131 dB PSNR |
| Text encoder, complete | Done, parity within 3e-08 |
| Diffusion transformer, complete | Done, parity within 3e-07 |
| Sampling loop and generation pipeline | Done |
| Execution layer: scan, fusion, residency, sharding, cache | Done |
| Notebook runner, ipywidgets and Gradio interfaces | Done |

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
src/
├── config/            configuration dataclasses and enumerations
│   ├── precision.py       NumericPrecision
│   ├── runtime.py         residency strategy, resolution buckets
│   ├── vae.py             decoder layer and structure settings
│   └── checkpoint.py      bundle source, top-level InferenceConfig
├── utils/             cross-cutting, no model knowledge
│   └── logging.py
├── checkpoint/        loading weights and addressing them
│   ├── hub.py             download from the Hub
│   ├── restore.py         restore Orbax pytrees
│   └── parameters.py      flat-key access helpers
├── layers/            individual mathematical primitives
│   ├── convolution.py
│   ├── normalization.py     group norm and RMS norm
│   ├── positional.py        rotary embedding, half-split pairing
│   ├── axial_positional.py  rotary embedding, interleaved, multi-axis
│   ├── embedding.py         sinusoidal timestep embedding
│   ├── masking.py           causal and padding attention mask
│   ├── resampling.py
│   └── activation.py
├── blocks/            composites assembled from primitives
│   ├── residual.py
│   ├── attention.py             autoencoder, single head, unmasked
│   ├── grouped_query_attention.py   text encoder, masked, rotary
│   ├── joint_attention.py           transformer, multi-head, unmasked
│   ├── feedforward.py
│   ├── gated_mlp.py
│   ├── modulation.py
│   ├── double_stream.py
│   ├── single_stream.py
│   └── transformer_layer.py
├── models/            complete networks assembled from blocks
│   ├── vae.py
│   ├── text_encoder.py
│   └── transformer.py
├── sampling/          rectified-flow sampler
│   ├── schedule.py        noise levels
│   ├── euler.py           the integration loop, stepped or fused
│   └── latent.py          spatial and token forms
├── execution/         placement and compilation, never semantics
│   ├── residency.py       which components stay in accelerator memory
│   ├── sharding.py        splitting parameters across devices
│   └── compilation.py     persistent compilation cache
├── telemetry/         run instrumentation, never semantics
│   ├── stages.py          blocking-aware stage timing and profiles
│   ├── arrays.py          tensor shape, dtype, size and placement
│   └── devices.py         platform and accelerator memory
├── tokenization/      prompt text to padded token identifiers
│   ├── fast.py            tokenizers plus Jinja, no deep learning framework
│   └── prompt.py          entry point, falls back to transformers
├── interfaces/        front ends, wiring only
│   ├── session.py         input handling both front ends share
│   ├── widgets.py         in-notebook controls
│   └── browser.py         Gradio interface
└── pipeline.py        end-to-end generation, the only stateful module

notebooks/
└── generate.ipynb     runner; contains no logic, only calls into src
```

Dependencies run strictly downward through that list: a layer may
import from config and utils but never from blocks; a block may import
from layers but never from models. Numeric code (layers, blocks,
models) is pure, taking arrays and a configuration object and
performing no IO or logging. Orchestration code (checkpoint) performs
IO and logs every stage.

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

Integration tests live in `tests/integration/` and are excluded from
that run because they require network access and, for parity testing,
PyTorch. PyTorch is used only to produce reference outputs and is never
a dependency of the `src` package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/black-forest-labs/flux2 /tmp/flux2
python -m tests.integration.test_vae_parity --reference-source-path /tmp/flux2
```

Tests come in two tiers. Smoke tests check shape, dtype and basic
execution. Regression tests check numerical correctness against
independently implemented oracles, swept across a matrix of shapes
generated at run time rather than compared against stored golden
arrays.

## Debugging

Per-stage timings, tensor placements and accelerator memory are logged
automatically. To also watch individual tensors from inside the compiled
model:

```python
from src.telemetry import enable_model_tracing, disable_model_tracing

enable_model_tracing("transformer")   # or "vae", "text_encoder", or "" for all
image = pipeline.generate(request)
disable_model_tracing()
```

Each trace point reports shape, dtype and value statistics on every
execution, including once per block inside a scanned stack. It is off by
default because each point is a host callback that serialises the
program; expect a generation to be several times slower while it is on.

Every timed stage also separates compilation from execution
automatically, which is what distinguishes a stage that will be fast on
a second run from one that will not:

```
  stage                            total   compile   share  detail
  vae decode                      13.66s     0.37s   74.4%  precision highest
  sampling                         4.71s     4.57s   25.6%  4 steps, fused, scanned blocks
  compilation is 27% of this run and is cached; a repeat should cost about 13.43s
```

## Contributing

Read [AGENTS.md](AGENTS.md) first. It documents the architectural
conventions, the verified model facts this implementation depends on,
and the hazards that have already caused bugs here.

## License

Apache-2.0.
