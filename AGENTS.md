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
flux2_klein/
├── logging_setup.py   # dual console/file logging
├── config.py          # every tunable value, as dataclasses
├── checkpoint.py      # download and restore the JAX-native bundle
└── layers.py          # VAE layer primitives (pure functions)
tests/
├── run_all_tests.py   # single entry point, writes a full log
├── test_config.py
├── test_checkpoint.py
└── test_layers.py
```

Dependency direction is strictly one-way: `layers` depends on `config`;
`checkpoint` depends on `config`; nothing depends on `layers` yet.
`config` depends on nothing. Do not introduce a cycle.

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

Related: the checkpoint bundle repository was private during early
development, so `checkpoint.py`'s download path is verified by **mocks
only**. Its end-to-end behaviour against the real Hub has not been
exercised in a sandbox. Treat the first real run as a validation step.

## 7. Why there is no logging inside numeric functions

Python-level logging inside a `jax.jit`-compiled function executes once
during tracing and never again during actual execution. A log line
inside a traced numeric kernel is not merely useless; it actively
misleads, appearing to report per-call behaviour while reporting a
single trace. Observability belongs at the orchestration layer:
checkpoint load, pipeline stage boundaries, and the test runner.

## 8. Current status and remaining work

Done:

- Configuration, residency strategy, resolution buckets
- Checkpoint download and per-component Orbax restore
- Dual console/file logging
- VAE layer primitives: convolution, group normalization,
  nearest-neighbour upsample, SiLU

Remaining, in intended order. Each phase should be finished and tested
before the next begins, rather than writing everything and debugging at
the end:

1. **VAE**: residual block, attention block (with query chunking, see
   below), full decoder assembly, parity against the reference,
   resolution scaling.
2. **Text encoder**: RMSNorm, RoPE, masking, GQA attention (32 query
   heads, 8 key/value heads), SwiGLU MLP, single layer, 27-layer stack,
   chat template and tokenization.
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
