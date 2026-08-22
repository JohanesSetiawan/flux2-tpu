# flux2-tpu

[![tests](https://github.com/JohanesSetiawan/flux2-tpu/actions/workflows/tests.yml/badge.svg)](https://github.com/JohanesSetiawan/flux2-tpu/actions/workflows/tests.yml)

JAX-native inference for [FLUX.2 Klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
on free-tier TPU. No PyTorch anywhere in the runtime.

Every component is verified numerically against the reference
implementation it was ported from: the autoencoder to 131 dB PSNR, the
text encoder and diffusion transformer to within float32 rounding.

> **Status: experimental.** The pipeline works end to end and its
> numerics are verified, but it has run on a small number of machines
> and the API is not yet stable.

> **No safety filtering and no watermarking.** This port includes
> neither, and the reference implementation ships both. See
> [Licensing and safety](#licensing-and-safety) before deploying this
> anywhere other people can reach.

## Quick start

Open [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb) in Kaggle:

1. Set the accelerator to **TPU VM v5e-8**
2. Attach the weights dataset
3. **Run All**

The last cell opens a web interface. Type a prompt, press Generate,
repeat as often as you like.

The first run compiles the model, which takes a few minutes. Later runs
reuse the compilation cache from `/kaggle/working` and start in seconds.

## Why Kaggle

Both platforms run the same code, but Kaggle is substantially better
suited to it:

| | Kaggle v5e-8 | Colab v5e-1 |
|---|---|---|
| TPU chips | 8 | 1 |
| Restoring 13 GB | ~9 s | ~7 min |
| Compilation cache | persists in `/kaggle/working` | lost with each VM |
| Weights | attachable as a dataset | downloaded each session |

The load-time gap is disk throughput, roughly 1450 MB/s against 30, and
is a property of the platforms rather than of this code. A Colab
notebook may follow; the package itself already runs there.

## Requirements

- TPU v5e or newer, or CPU for testing
- Python 3.10+
- `jax`, `orbax-checkpoint`, `numpy`, `tokenizers`, `jinja2`

Optional: `gradio` and `pillow` for the web interface, `transformers`
only as a tokenizer fallback, `torch` only for parity tests.

## Usage

```python
from pathlib import Path

from src.config import CheckpointSourceConfig, ExecutionConfig, InferenceConfig
from src.interfaces.session import build_request
from src.pipeline import Pipeline
from src.utils import configure_logging

logger = configure_logging(Path("run.log"), "flux2_klein")

pipeline = Pipeline(
    InferenceConfig(
        checkpoint_source=CheckpointSourceConfig(
            local_cache_directory=Path("/path/to/weights"),
        ),
    ),
    logger,
    execution_config=ExecutionConfig(
        compilation_cache_directory=Path("compilation_cache"),
    ),
)
pipeline.load()
pipeline.warm_up()

image = pipeline.generate(
    build_request(
        prompt="a lighthouse on a rocky shore at dusk",
        resolution_label="1024x1024",
        requested_seed=-1,
        buckets=pipeline.resolution_buckets,
    )
)
```

`generate` returns a float array in unit range, shaped
`(height, width, 3)`.

## Supported resolutions

| Resolution | Aspect | Image tokens |
|---|---|---|
| 1024 x 1024 | 1:1 | 4096 |
| 1360 x 768 | 16:9 | 4080 |
| 768 x 1360 | 9:16 | 4080 |

Three, deliberately. Past roughly 4300 image tokens the reference
sampling schedule switches to a formula derived for a fifty-step model,
and this checkpoint takes four steps. Exceeding that boundary is a
quality decision rather than a memory one.

## How it works

FLUX.2 Klein-4B is a four-step distilled rectified-flow model. Three
components run in sequence:

1. **Text encoder** (Qwen3-4B, 27 of 36 layers) turns a prompt into a
   7680-wide conditioning tensor
2. **Diffusion transformer** (5 double-stream and 20 single-stream
   blocks, hidden size 3072) predicts a velocity field, integrated over
   four Euler steps
3. **Autoencoder decoder** turns the resulting latent into an image

The step count and guidance value are fixed by distillation and are not
tunable; the reference implementation rejects attempts to change them.
Speed comes from compilation and placement, never from altering the
sampling mathematics.

**Precision.** Nothing is quantized. The transformer runs in bfloat16,
which is the checkpoint's own dtype, and the decoder in float32,
matching the reference decode path.

**Placement.** Parameters are split or replicated across the device
mesh per component. Splitting divides memory across a pod; replication
multiplies it, which is why the text encoder's layer stack is split
rather than copied.

## Architecture

```
src/
├── config/          configuration dataclasses; no literal lives outside here
├── utils/           cross-cutting, no model knowledge
├── checkpoint/      loading weights and addressing them
├── telemetry/       timing, placement, compilation and cache reporting
├── layers/          individual mathematical primitives
├── blocks/          composites assembled from primitives
├── models/          complete networks assembled from blocks
├── sampling/        the rectified-flow sampler
├── tokenization/    prompt text to padded token identifiers
├── execution/       placement and compilation, never semantics
├── interfaces/      front ends; wiring only
└── pipeline.py      end-to-end generation, the only stateful module
```

Dependencies run one way down that list. Numeric code is pure: arrays
and a config in, arrays out, no IO and no logging. Orchestration
performs IO and logs every stage.

## Testing

```bash
pip install -e .
JAX_ENABLE_X64=1 python -m tests.run_all_tests   # 202 tests
```

The 64-bit flag is not optional for the regression suites: they compare
against float64 oracles, and without it they would compare float32
against float32 and prove much less.

Three kinds of test, answering different questions:

- **Smoke** tests check shape, dtype and that the thing runs.
- **Regression** tests check numerical correctness against independently
  implemented oracles, or against properties that must hold. Inputs are
  generated at run time from seeded shapes rather than compared to
  stored golden arrays, so a test cannot pass by matching a value that
  was wrong when it was recorded.
- **Precision** tests run at bfloat16, the dtype production actually
  uses, because a suite running entirely in float64 cannot see a dtype
  promotion. One reached production that way.

Five [integration tests](tests/integration/) need network access and,
for parity, PyTorch. They are excluded from the unit suite and run
explicitly.

### What CI does and does not establish

The badge above means the unit suite passes on CPU. That covers logic:
layer mathematics, attention and rotary conventions, placement
arithmetic across simulated devices, tokenizer output, and dtype
preservation.

It does not mean a change works on TPU. Three classes of problem have
reached production here despite a passing suite, and none can appear on
CPU: allocation failures, which need real accelerator memory;
compilation cost, which dominates in practice and is not measurable
here; and precision modes, which are bit-identical on CPU and differ on
TPU.

It is a gate against regression in what already works, not evidence
that something new is correct.

## Observability

Every run reports a timing breakdown separating compilation from
execution, tensor placement per device, and whether the compilation
cache was reused:

```
  stage                            total   compile   share  detail
  vae decode                      13.66s     0.37s   74.4%  precision highest
  sampling                         4.71s     4.57s   25.6%  4 steps, fused, scanned blocks
  compilation is 27% of this run and is cached; a repeat should cost about 13.43s
```

To trace individual tensors from inside the compiled model:

```python
from src.telemetry import enable_model_tracing, disable_model_tracing

enable_model_tracing("transformer")   # or "vae", "text_encoder", "" for all
image = pipeline.generate(request)
disable_model_tracing()
```

Each trace point reports shape, dtype and value statistics on every
execution, including once per block inside a scanned stack. Off by
default: each point is a host callback that serialises the program.

## Contributing

Read [AGENTS.md](AGENTS.md) first. It records the architectural
conventions, the verified model facts this implementation depends on,
and the hazards that have already caused bugs here.

## Acknowledgements

- [Black Forest Labs](https://github.com/black-forest-labs/flux2) for
  FLUX.2 and the reference implementation
- [Comfy-Org](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b)
  for the float32 autoencoder weights matching reference decode
  precision

## Licensing and safety

### Licences

Two separate licences apply, and they are not the same thing:

| | Licence | Commercial use |
|---|---|---|
| This code | [Apache-2.0](LICENSE) | Permitted |
| [FLUX.2 Klein-4B weights](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 | Permitted |

Black Forest Labs released the 4B checkpoint under Apache-2.0
explicitly for commercial use. Note that this does **not** extend to
the whole FLUX.2 Klein family: the 9B checkpoints are released under a
non-commercial licence, and this repository targets 4B only.

The weights are not redistributed here. They come from the upstream
repository, whose model card also lists
[out-of-scope uses](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B#out-of-scope-use)
that apply regardless of the licence.

### What this port leaves out

The reference implementation ships two things this one does not. Both
omissions are deliberate, and both matter more if you deploy this than
if you run it yourself.

**Content filters.** The reference repository includes NSFW and
protected-content filters for both inputs and outputs. Black Forest
Labs describes these as required for the 9B models and *encouraged* for
4B. This port has none: no prompt filtering, no output classification,
nothing.

**Watermarking.** The reference includes pixel-layer watermarking and
documents C2PA metadata for content provenance, so generated images can
be identified as synthetic. Neither is implemented here. Images this
produces carry no marker of any kind.

Neither was omitted because it was judged unnecessary. They were out of
scope for a port whose goal was numerical parity with the reference
model, and adding them is a separate piece of work rather than a
setting to switch on.

If you are running this yourself for research, that is probably fine.
If you are putting it in front of other people, understand that you are
deploying a generative model with no safeguards at all, and that the
upstream authors ship safeguards they encourage you to use.

The model's own safety training remains intact: the checkpoint went
through the pre-training and post-training mitigations described in the
[model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
That is a property of the weights, not of this code, and it is not a
substitute for a filter.
