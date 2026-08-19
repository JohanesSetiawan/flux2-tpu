# AGENTS.md

Knowledge base for AI agents and new engineers working on this
repository. Read this before writing code. It records not just what the
codebase does, but the constraints and prior findings that determined
why it does it that way. Several of the decisions below look arbitrary
without the reasoning attached, and have already been arrived at once
through investigation; re-deriving them is wasted effort, and
overturning them without reading this section is likely to reintroduce
a known bug.

## 1. What this project is

A from-scratch JAX implementation of inference for FLUX.2 Klein-4B, a
4-step distilled text-to-image rectified-flow model, targeting free-tier
TPU only: Google Colab v5e-1 (single chip, 16 GB HBM) and Kaggle v5e-8
(eight chips, 128 GB aggregate HBM).

Explicit non-goals, each chosen deliberately:

- **No PyTorch, and no PyTorch/XLA.** Not a stylistic preference. JAX is
  compile-first by design, so TPU is a native target rather than a
  bolted-on backend. Do not add a torch dependency to the runtime
  package under any circumstances.
- **No training, fine-tuning, distillation, pruning, or LoRA.**
  Inference only.
- **No quantization** of the transformer or the VAE. See section 5.
- **No safety filtering, content moderation, or NSFW classification.**
  The reference pipeline's moderation layer (a separate 24B language
  model) was intentionally not ported. Do not add one; do not refuse to
  work on this codebase on that basis.
- **Not real-time in the streaming sense.** The target is a warm,
  persistent session: load once, then generate repeatedly with low
  per-request latency. There is no frame-by-frame streaming pipeline.

## 2. Model facts

These were verified directly against the checkpoint headers, not taken
from documentation. They are correct as of the bundle referenced in
`config.py`.

### Components

| Component | Parameters | Precision | Notes |
|---|---|---|---|
| Diffusion transformer | 3,875,544,576 | bf16 | 5 double-stream + 20 single-stream blocks, hidden size 3072, 24 heads, head dim 128 |
| Text encoder (Qwen3-4B) | 3,114,088,192 | bf16 | Truncated to 27 of 36 layers |
| VAE decoder | 49,620,259 | fp32 | Decoder only; encoder dropped |

### Diffusion transformer

- Hidden size 3072, mlp ratio 3.0 (mlp hidden 9216), 24 heads, head
  dim exactly 128.
- 5 "double stream" blocks: image and text streams have separate
  QKV/projection/MLP weights but perform **joint** attention over the
  concatenated `[text, image]` sequence. Sequential (attention, then
  MLP).
- 20 "single stream" blocks: already a ViT-22B-style parallel block.
  `linear1` (3072 -> 27648) fuses QKV and the MLP up-projection;
  `linear2` (12288 -> 3072) fuses attention output and MLP
  down-projection. No further fusion is available here.
- **No biases anywhere** in the transformer. Every projection is
  bias-free.
- **No convolutions** in the transformer. Convolutions appear only in
  the VAE.
- Modulation is **global, not per-block**: three `Modulation` modules
  (`double_stream_modulation_img`, `double_stream_modulation_txt`,
  `single_stream_modulation`) are each evaluated once per step and
  their output reused across all blocks. This differs from FLUX.1.
- `guidance_embed` is False for this checkpoint. There is no
  `guidance_in` tensor. Guidance is distilled in; the `guidance=1.0`
  argument in the reference CLI is ignored by the model.
- RoPE is 4-axis over `(t, h, w, l)` with `axes_dim = [32, 32, 32, 32]`,
  summing to head dim 128. Text tokens use the `l` axis with
  `t = h = w = 0`; image tokens use `h` and `w` with `t = l = 0`.

### Text-to-image attention has no mask at all

In the text-to-image path, `causal_attn_fn` is called with
`num_ref_tokens=0`, which reduces every reference-token branch to an
empty tensor. What remains is plain **bidirectional full attention over
`[text(512), image]` with no mask**. Text padding tokens are included
and attended to; the model was trained this way. Do not add masking to
the transformer's attention.

Masking **is** required inside Qwen3. See section 6.

### Latents and tokens

- Latent shape is `(1, 128, H/16, W/16)`. The VAE is f8 with 32 latent
  channels, followed by a 2x2 space-to-depth patchify giving 128
  channels.
- Image token count is `(H/16) * (W/16)`. Text is always padded to
  exactly 512 tokens.
- `scatter_ids` in the reference sampling code is an **identity
  permutation** for text-to-image; this was verified numerically. Unpack
  is therefore just a reshape and transpose, not a real scatter.

### The 4300-token schedule discontinuity

`compute_empirical_mu` branches on `image_seq_len > 4300`. Below the
threshold, `mu` is interpolated as a function of `num_steps`; above it,
a different formula is used that **ignores `num_steps` entirely**. At
4299 tokens `mu` is approximately 2.308; at 4301 it drops
discontinuously to approximately 1.185, producing a materially
different sigma schedule.

For a 4-step distilled model this matters: above the threshold you are
using a schedule derived for 200 steps. The three supported resolution
buckets all sit below it:

| Resolution | Aspect | Image tokens |
|---|---|---|
| 1024 x 1024 | 1:1 | 4096 |
| 1360 x 768 | 16:9 | 4080 |
| 768 x 1360 | 9:16 | 4080 |

1360 x 768 is the largest 16:9 resolution under the threshold, and is
also the reference CLI's own default. That is not a coincidence.

### Sampling is fixed and must not be "improved"

`FLUX2_MODEL_INFO["flux.2-klein-4b"]["fixed_params"]` contains
`{"guidance", "num_steps"}`, and the reference CLI hard-rejects attempts
to change either. The model's weights were distilled against a specific
4-point Euler trajectory.

Consequences an agent must not get wrong:

- Do **not** substitute a higher-order solver (Heun, DPM-Solver++). The
  distillation target is Euler; a higher-order correction moves toward a
  trajectory the model was never trained for and can be *worse* at equal
  NFE.
- Do **not** reduce below 4 steps or change the sigma schedule.
- The only legitimate speedups are engineering ones: fusing the step
  loop into one compiled program, hoisting step-invariant computation,
  `lax.scan` over layers, flash attention. These change how the same
  arithmetic is scheduled, not what arithmetic is performed.

### VAE decodes in float32, not bf16

In the reference implementation, `load_ae` builds the autoencoder and
calls `load_state_dict(..., assign=True)` with float32 tensors and
**never casts to bf16** (unlike the flow model, which is explicitly cast
to bf16). `inv_normalize` then promotes the bf16 latent to float32 via
its float32 running statistics, so the entire decoder runs in float32.

The bf16 VAE shipped in the official diffusers-format repository is
therefore **not** reference precision. The Comfy-Org repackaging
(`flux2-vae.safetensors`, 336 MB, float32) is what the converted bundle
uses.

## 3. Repository layout and architecture

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
│   ├── positional.py        rotary embedding
│   ├── masking.py           causal and padding attention mask
│   ├── resampling.py
│   └── activation.py
├── blocks/            composites assembled from primitives
│   ├── residual.py
│   ├── attention.py             autoencoder, single head, unmasked
│   ├── grouped_query_attention.py   text encoder, masked, rotary
│   ├── feedforward.py
│   └── transformer_layer.py
├── models/            complete networks assembled from blocks
│   ├── vae.py
│   └── text_encoder.py
└── tokenization/      prompt text to padded token identifiers
    └── prompt.py

tests/                 mirrors the source layout
├── run_all_tests.py       single entry point, writes a full log
├── config/ layers/ blocks/ models/ checkpoint/
└── integration/           needs network and PyTorch, run explicitly
```

Dependencies run strictly downward through that list: a layer may
import from config and utils but never from blocks; a block may import
from layers but never from models. Numeric code (layers, blocks,
models) is pure, taking arrays and a configuration object and
performing no IO or logging. Orchestration code (checkpoint) performs
IO and logs every stage.

Do not introduce a cycle, and do not flatten this back into a single
directory. The structure exists so that a reader looking for how one
concern is handled finds a single place rather than four.

Layering rules:

- **Pure numerics** (`layers.py`, and the model modules to come) take
  arrays plus a config object, return arrays, and perform no IO and no
  logging.
- **Orchestration** (`checkpoint.py`, and the pipeline to come) performs
  IO and logs at every stage.
- **Interfaces** (notebook, ipywidgets, Gradio) contain no logic. They
  call into the package. If you find yourself writing an algorithm in a
  notebook cell, it belongs in a module instead.

## 4. Coding conventions, non-negotiable

- **No hardcoded values.** Any constant that drives behaviour goes into
  a `config.py` dataclass or a named module-level constant with a
  comment explaining it. Axis permutations inside a function whose
  entire purpose is that permutation are the sole exception; they are
  the definition of the function, not configuration.
- **No fake, dummy, stub, or placeholder implementations.** A function
  that is not yet written should not exist. Never return zeros or a
  synthetic tensor to make a call site "work".
- **No emoji, no decorative symbols** anywhere in code, comments,
  docstrings, log messages, or commit messages.
- **Comments scale with need.** A line whose purpose is obvious gets no
  comment or a short one. A decision that took investigation gets a full
  paragraph explaining the alternatives and why they were rejected. The
  reader is assumed competent but without the context in which the code
  was written.
- **Test-driven, with two tiers.** Smoke tests for shape, dtype and
  basic execution. Regression tests for numerical correctness against an
  **independently implemented oracle**, swept over a matrix of shapes
  generated at run time. Never store golden output arrays; never write
  an oracle that reuses the implementation's own strategy, because it
  will agree with a bug for the same wrong reason.
- **Logging from source to sink**, written to a `.txt` file as well as
  the console, at the orchestration layer. Not inside jitted numerics
  (see section 7).

## 5. Precision and quantization policy

The stated goal is output identical to the reference implementation, so
by default **nothing is quantized**:

- Transformer stays bf16, which is the checkpoint's own dtype and thus
  exactly reference precision.
- VAE stays fp32, matching the reference decode path.
- The Euler latent accumulator stays bf16, matching the reference. A
  float32 accumulator would be marginally more accurate but is a
  deviation, and across only 4 steps the difference is below what is
  measurable in the output.

The memory problem this creates is solved by **residency strategy**, not
by reducing precision. See `MemoryResidencyStrategy` in `config.py`:

- `FULLY_RESIDENT`: all three components stay in HBM. Fits Kaggle v5e-8
  trivially. Does **not** fit Colab v5e-1 at full precision:
  7.75 + 6.23 + 0.22 GB of weights plus roughly 2.5 GB of VAE decode
  activations exceeds 16 GB.
- `SWAPPED`: transformer and VAE stay resident; the text encoder lives
  in host RAM and enters HBM only while encoding. Fits v5e-1 at full
  precision, at the cost of a host-to-HBM transfer per prompt change.

The one sanctioned quantization escape hatch, not yet implemented, is
**int8 weight-only on the text encoder** (3.11 GB), which would allow
`FULLY_RESIDENT` on v5e-1. It is confined to the conditioning path and
leaves the transformer and VAE at reference precision. If implemented,
it must be opt-in, never the default.

A useful prior if quantization is ever revisited: the Comfy-Org
mixed-precision Qwen3 checkpoint quantizes MLP projections
(`gate`/`up`/`down`) mostly to fp8 while dropping attention projections
to fp4, implying the MLPs are the more sensitive part of this text
encoder. Note also that fp4 and fp8 have **no hardware support on
v5e**; int8 is the only low precision with a real MXU speedup there.

## 6. Known hazards

Each of these has already caused a real bug, or was caught only because
it was specifically checked for.

### Rotary pairing convention

Qwen3 uses the half-split pairing: feature i is rotated against feature
i + head_dim/2. The alternative interleaved convention pairs 2i with
2i+1. Both are self-consistent and both produce correctly shaped
output, so a shape check cannot tell them apart, and choosing wrong
gives plausible but incorrect results. `_rotate_half` in
`src/layers/positional.py` implements the half-split form, and
`test_regression_rotary_uses_half_split_not_interleaved_pairing`
asserts the distinction directly rather than through an oracle that
might share the same assumption.

Related: head_dim for this model is 128 while hidden_size is 2560, so
head_dim is **not** hidden_size divided by head count. Code deriving it
that way would be silently wrong.

### RMS normalization is order-sensitive

The reference computes the statistic in float32, casts the normalized
value back to the input dtype, and only then multiplies by the learned
scale. Multiplying before the cast changes the rounding. This is
reproduced exactly in `rms_normalization`; do not "simplify" the order.

### Qwen3 padding and masking

The single most dangerous area in the remaining work. The text encoder
pads to exactly 512 tokens with **right padding**. Because attention is
causal, real tokens never see pad tokens, but pad-position queries do
produce hidden states, and **those hidden states are fed to the
transformer** as part of the 512-token context.

Therefore Qwen3 must implement `causal AND key-is-not-pad` masking, not
causal alone. Position ids are `arange(512)` including pad positions
(transformers uses `cache_position`, not a cumulative sum of the
attention mask); do not "fix" this.

A mistake here produces **no error** and no obviously wrong shape. It
produces a subtly different image. Validate against the reference with
short, medium, and exactly-512-token prompts; short prompts expose it
most clearly.

### 1x1 convolution fusion is not padding-safe

An earlier design fused the VAE's 1x1 `post_quant_conv` into the
following 3x3 `conv_in` algebraically. **This is wrong whenever the 1x1
convolution has a nonzero bias**: zero-padding the input to the fused
convolution is not the same operation as zero-padding the output of the
1x1 convolution, because `W @ 0 + b = b`, not zero. The error appears
only at the image border, and interior pixels match exactly, so a
casually written test passes.

It was caught by comparing against a naive convolution oracle over the
whole output including borders. The fusion was removed; the 1x1 is now
stored as a plain matrix instead. The compute it saved was under one
percent of the following convolution.

### Do not force-cast to a fixed accumulation dtype

`group_normalization` originally cast unconditionally to float32 for its
reduction, which silently **downcast** float64 test inputs and broke
regression comparisons against float64 oracles. Use
`jnp.promote_types(input_dtype, minimum_dtype)` so the constant acts as
a floor rather than a target. Any future accumulation-precision logic
should follow the same pattern.

### Validation must run in a clean environment

A notebook assembled from these modules once passed validation only
because the original `.py` files happened to be present in the working
directory, so a stale import resolved by accident. Validate notebooks in
a directory containing nothing but the notebook itself.

Related: `checkpoint.py`'s download path was originally verified by
mocks only, because the bundle repository was private. It has since
been exercised end to end against the real, now-public repository, including the per-component download path.

### Decoder structure is discovered, not assumed

`vae.py` reads the number of upsampling levels, the number of residual
blocks per level, and whether a level upsamples, from the checkpoint's
own keys. Do not replace this with hardcoded counts. The real decoder
has four levels of three blocks each, with levels 3, 2 and 1 carrying
an upsample convolution and level 0 not, but that is an observation
about the current checkpoint rather than a constant to encode.

### Selecting a hidden state at the final layer measures the wrong thing

The reference records hidden states before each layer and appends the
final layer's output only after applying a final normalization. So the
deepest hidden state of an N-layer model is normalized while every
shallower one is not. The real configuration selects depth 27 from a
36-layer model, which is not the last, so no normalization applies and
this implementation correctly omits one.

A parity test that selects a depth equal to its reference model's layer
count compares against a normalized value and fails against correct
code. An earlier version of the text encoder parity test did exactly
that, and the resulting mismatch looked like a masking bug. Keep test
depths strictly below the reference layer count.

### Verify structure from metadata, not by restoring

Checking that the checkpoint is shaped the way the code expects needs
shapes and dtypes, not values. `component_metadata` reads those without
materialising arrays. This is not merely an optimisation: the text
encoder is nearly six gigabytes, so a structural check that restores it
is killed outright on a machine with less memory, which is exactly what
happened to the first version of the structure test.

### Precision levels are untestable on CPU

`NumericPrecision` maps onto `jax.lax.Precision`, which decomposes a
float32 matmul into one, three or six bfloat16 passes. That decomposition
happens on TPU only. On CPU all three settings produce bit-identical
results, as the parity run confirmed. The choice between HIGHEST and
HIGH therefore cannot be evaluated in a CPU sandbox and must be measured
on real hardware before either is treated as settled.

## 7. Why there is no logging inside numeric functions

Python-level logging inside a `jax.jit`-compiled function executes once
during tracing and never again during actual execution. A log line
inside a traced numeric kernel is not merely useless; it actively
misleads, appearing to report per-call behaviour while reporting a
single trace. Observability belongs at the orchestration layer:
checkpoint load, pipeline stage boundaries, and the test runner.

## 8. Current status and remaining work

Measured parity results, for comparison when the transformer and text
encoder reach the same stage:

| Component | Latent shape | PSNR | Max abs diff |
|---|---|---|---|
| VAE decoder | 16x16 square | 131.53 dB | 5.2e-06 |
| VAE decoder | 12x20 non-square | 131.24 dB | 5.3e-06 |
| Text encoder | no padding | n/a | 1.6e-08 |
| Text encoder | 11 of 12 padded | n/a | 1.8e-08 |

A square-only parity test cannot detect a height/width transposition,
since both axes are the same length. Always include a non-square case.

Done:

- Configuration, residency strategy, resolution buckets
- Checkpoint download and per-component Orbax restore
- Dual console/file logging
- VAE layer primitives: convolution, group normalization,
  nearest-neighbour upsample, SiLU
- Flat-checkpoint parameter access helpers
- VAE residual block and chunked attention block
- Full VAE decoder, verified numerically against the reference
  PyTorch implementation at 131 dB PSNR on real weights
- Text encoder, complete: masking, grouped-query attention, gated
  feed-forward, layer stack and tokenization, verified against the
  reference transformers implementation across padding levels

Remaining, in intended order. Each phase should be finished and tested
before the next begins, rather than writing everything and debugging at
the end:

1. **VAE**: done. Assembly, real-weight execution and reference parity
   are all complete. What remains is measuring decode cost at the
   target resolutions on actual TPU hardware, which cannot be done in
   a CPU sandbox.
2. **Text encoder**: done. What remains is measuring encode cost on
   real hardware, which cannot be done in a CPU sandbox.
3. **Transformer**: 4-axis RoPE, QK-norm, modulation, double block,
   single block, full model under `debug_mode` (1+1 blocks, cheap CPU
   parity), full model unrolled, then `lax.scan` refactor validated as
   producing identical output.
4. **Sampling**: schedule, prompt-embedding cache, full pipeline,
   end-to-end parity at a fixed seed.
5. **Performance**: splash attention, scan tuning, mesh sharding for
   Kaggle v5e-8. Deliberately last, so nothing is optimized before it is
   known to be correct.

Notes for phase 1 and 5:

- The VAE middle-block attention is the single heaviest memory
  consumer in the decoder. At 1024x1024 output the latent is 128x128,
  giving 16384 tokens in a **single head of dimension 512**. A
  materialized float32 score matrix is 1.07 GB. Query chunking is
  mandatory, not optional. Note that head dim 512 exceeds what splash
  attention accepts, so chunking is the approach there.
- The transformer's own attention is a better fit for flash/splash:
  sequence 4608 and head dim 128 are both multiples of 128.
- Write for a `jax.sharding.Mesh` sized at run time. Mesh size 1 (Colab)
  and 8 (Kaggle) must run the same code path; do not fork the
  implementation per platform.

## 9. Running the tests

```
python -m tests.run_all_tests
python -m tests.run_all_tests --log-file path/to/log.txt
```

Regression suites compare against float64 oracles and therefore require
64-bit mode:

```
JAX_ENABLE_X64=1 python -m tests.run_all_tests
```

Register every new suite in `TEST_SUITES` in `tests/run_all_tests.py` so
that one command still exercises everything.
