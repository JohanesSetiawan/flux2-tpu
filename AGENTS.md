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

### There are two rotary conventions in this codebase, and they differ

This is now the highest-risk confusion in the repository. Both models
use rotary position embedding, under **incompatible pairing
conventions**:

| Model | Module | Pairing |
|---|---|---|
| Text encoder | `src/layers/positional.py` | half-split: feature i with i + head_dim/2 |
| Diffusion transformer | `src/layers/axial_positional.py` | interleaved: feature 2i with 2i+1 |

Do not merge these modules, and do not "simplify" one into the other.
Both are self-consistent, both produce correctly shaped output, and
each matches only its own checkpoint. Each has a dedicated test pinning
its convention by placing a unit value at one feature and checking
which index receives the rotated component.

The transformer's version additionally carries four independent
position axes, with the head dimension partitioned between them. A
consequence that has already caused a test failure: a feature is
rotated only by its own axis's position. Setting a position on axis 1
leaves feature 0 untouched, because feature 0 lies in axis 0's slice.

For text-to-image, text tokens carry their sequence index on the last
axis and images carry row and column on the middle two, so the two
groups never collide despite sharing one unmasked attention sequence.
The first axis is unused in this mode.

### Text encoder rotary pairing convention

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

### There are three attention implementations and none is interchangeable

| Model | Module | Shape |
|---|---|---|
| Autoencoder | `blocks/attention.py` | single head, head dim equals channel count, unmasked, query-chunked |
| Text encoder | `blocks/grouped_query_attention.py` | multi-head, fewer key/value heads, causal plus padding mask |
| Transformer | `blocks/joint_attention.py` | multi-head, fully unmasked over concatenated text and image |

The transformer's lack of any mask is deliberate and matches the
reference: every token attends to every other, including the text
encoder's padding positions. Adding a mask to "fix" that would diverge
from the trained model.

### Layer normalization hides constant perturbations

A test that perturbs activations by adding the same constant to every
feature will see no effect anywhere downstream of a layer
normalization, because that operation subtracts the mean. This is not a
bug and not a sign the value is unused. Perturb with non-uniform noise
instead. One transformer test made this mistake and appeared to show
that the text stream did not influence the image stream.

### The sampler is verified against an analytic solution, not an oracle

`test_regression_euler_integrates_a_known_field_correctly` feeds the
integrator a velocity field whose exact solution is known, rather than
comparing against another implementation. It asserts two things: that
the result matches explicit Euler exactly, and that refining the steps
converges toward the analytic answer. The second is what catches a sign
error, which would satisfy the first while integrating backwards.

### Replication multiplies memory across a pod, it does not divide it

The mistake that exhausted an eight-chip v5e-8 before a single image was
generated. `plan_component_residency` was reasoned about as "128 GiB
aggregate, everything fits", but aggregate capacity is meaningless for a
replicated component: what is available per chip stays 15.75 GiB, and a
replicated 5.80 GiB text encoder weighs 5.80 GiB on every one of them,
plus 46 GiB of host transfer during load.

`SPLITTABLE_GROUPS_BY_COMPONENT` in `src/execution/sharding.py` now
decides per component and per group. The text encoder's layer stack is
split, bringing it from 5.80 GiB per chip to 0.63 GiB; its embedding
table stays replicated because it is read by gather, and splitting a
lookup table would make every lookup a collective.

Measured per-device weight, computed from each array's actual sharding
rather than from tree totals, is now logged at load. Report a
component's total size alone and an over-replication looks affordable
when it is not.

When adding a component, add it to the policy table. An absent one
replicates everything, which is safe but wasteful; a component never
placed at all stays on the first device while the rest of the pod
idles.

### Every execution option must be output-preserving, and is tested as such

`src/execution` and the `ExecutionConfig` flags change how work is
placed and compiled, never what is computed. That is asserted directly:
a scanned block stack against an unrolled one, and a fused sampling
loop against a stepped one. Anything that changed results would belong
in a model configuration instead.

Neither pair agrees bitwise, and the expected magnitudes differ for a
reason worth knowing. Both models carry float32 rotary tables, so
reordering operations around them amplifies float32 rounding and the
gap lands near 1e-8. Rebuilding those tables in float64 drops it to
about 1e-15, which is how the difference was confirmed as precision
rather than logic. The sampling loop carries no such table, so its two
paths agree to 1e-16.

### Eviction only frees memory if the reference is dropped

`evict_to_host` returns a new tree; the accelerator copy stays alive
until the caller drops its reference to the old one. Holding both is
the most common way this optimisation silently does nothing. The
pipeline deletes its reference explicitly after each use for that
reason.

### Sharding is tested across simulated devices

The execution suite sets `--xla_force_host_platform_device_count`
before importing JAX, so sharding runs across several devices rather
than the one a CPU really has. Testing it on a single device would
exercise only the trivial path. The suite asserts the device count took
effect, because a silent fallback to one device would make every
sharding test vacuous.

### Interfaces hold no logic, and that is enforced by where the tests are

`src/interfaces/session.py` holds every decision the front ends make:
resolving a label to a bucket, resolving a seed, building a request,
converting an image. The widget and Gradio modules are wiring. That
split exists so the behaviour is testable without clicking anything,
and the interface suite tests `session` thoroughly while asserting only
one thing about the front ends themselves: that importing them does not
require their optional toolkit.

If a front end starts making decisions, move them to `session` rather
than testing the front end.

### One dtype must govern the whole transformer forward pass

Found on the first real TPU run, and invisible before it. The timestep
embedding was built in float32 regardless of its input, so against
bfloat16 weights the modulation vectors came out float32, promoted the
activations they scaled, and left the text stream entering a block as
bfloat16 and leaving it float32.

A Python loop tolerates that silently. A scan does not: its carry input
and output dtypes must match exactly, so the program failed to compile
with a carry type error rather than running slowly.

Two things now prevent it. `timestep_embedding` returns its input's
dtype, matching the reference, which ends with a cast to `t`. And
`predict_velocity` derives a single compute dtype from the latent and
casts every entry point to it, so mixed inputs cannot promote the
residual stream partway through a block.

This class of bug cannot be caught by float64 tests, because float64
everywhere hides the promotion. The regression tests for it run at
bfloat16 deliberately.

### Notebook bootstrap must survive a kernel restart

A restart resets the working directory and clears `sys.path`, so a cell
that only ran `%cd` during the first pass leaves `src` unimportable
afterwards, and the apparent fix is to restart the whole session and
re-download everything. The bootstrap cell is therefore idempotent:
clone only if missing, always set both the directory and the import
path, and import `src` immediately so a failure surfaces there rather
than three cells later.

### Debugging inside compiled code needs jax.debug, not print

A Python `print` inside a jit region executes once, while the function
is traced, and never again during the thousands of times the compiled
program runs. It looks like per-call reporting and is not.

`src/telemetry/tracing.py` uses `jax.debug.print`, which emits a real
operation calling back to the host on every execution and survives into
a `lax.scan` body, so a scanned twenty-block stack reports twenty times
rather than once. Trace points are already placed throughout the three
models.

Turn it on with `enable_model_tracing("prefix")` and off with
`disable_model_tracing()`. Both clear JAX's compilation caches, and they
must: trace points are resolved at trace time, so a program already
compiled with tracing off would otherwise keep it off and the feature
would appear broken.

It is off by default because every trace point is a host callback, and
callbacks serialise the program around them. Expect a generation to be
several times slower with tracing on. Use the prefix filter: tracing
everything in a twenty-block stack buries the line that matters.

### Model entry points must be wrapped in jax.jit

Without an enclosing jit, every operation inside a model becomes its own
compiled program. This is easy to miss because the results are correct
either way; only the cost differs, and it differs enormously.

Measured on a v5e-1 before the fix: one autoencoder decode compiled
thirty-five separate programs taking eighty-seven seconds, of which
seventy-eight was `jit(conv_general_dilated)`, the convolutions compiled
one at a time. The text encoder compiled fifty-one programs including
`jit(cos)` and `jit(sin)` individually.

`Pipeline` now wraps `encode_prompt`, `predict_velocity` and
`decode_latent` in `jax.jit` with the configuration objects and latent
dimensions as static arguments, which is valid because every
configuration here is a frozen dataclass and therefore hashable.

Any new model entry point must be wrapped the same way. The symptom to
watch for in a profile is a compilation list naming primitive
operations rather than one entry per model.

### Measured results of the jit fix, for calibration

Wrapping the model entry points in jax.jit, measured on Colab v5e-1 at
1024x1024:

| | before | after |
|---|---|---|
| decode programs compiled | 35 | 17 |
| decode, steady state | 1.65s | 0.53s |
| decode compile per call | 0.80s | none |
| generation, new prompt | 6.10s | 4.85s |
| generation, cached prompt | 3.70s | 2.31s |

Note what did **not** move: the one-off decode compile stayed near 84
seconds. It is now a single `jit(decode_latent)` rather than thirty-five
fragments, but XLA still needs that long to compile a full-resolution
convolutional decoder in float32. That cost belongs to the persistent
compilation cache, not to further restructuring, which makes putting
the cache somewhere durable the highest-value remaining change on
Colab.

### Copy whole directories from upstream, never a list of filenames

The converter that produced the bundle named the six tokenizer files it
wanted. The upstream repository has seven, and the missing one,
`tokenizer.json`, is the only one that defines the tokenization pipeline
completely.

Nothing failed. The loader in use at the time could rebuild from
vocabulary and merges, so the omission stayed invisible for months, and
surfaced only when a faster tokenizer needing that file could not be
used on a bundle whose source had shipped it all along.

The directory is 15.9 MB against a bundle of 11 GB. Selecting files to
save 11 MB at the risk of dropping an important one is a bad trade at
any ratio. More importantly, a hand-written list encodes today's
implementation choices into the artifact: whoever wants a different
loader later finds the bundle already decided against them.

The same applies to any directory a future bundle carries. The
converter itself lives outside this repository, but this is the one
decision from it worth carrying forward.

### The chat template lives in one of two places

Upstream repositories disagree. Some embed the template in
`tokenizer_config.json` under a `chat_template` key; others ship a
separate `chat_template.jinja` and omit the key. FLUX.2 Klein does the
latter, Qwen3 the former, and both were verified byte-identical.

`_read_chat_template` checks both. An earlier version checked only the
embedded key and fell back to the slow tokenizer on a bundle that had
everything it needed.

### The fast tokenizer must stay identical, and is tested that way

`src/tokenization/fast.py` replaces the transformers tokenizer with the
Rust tokenizers library plus Jinja, cutting the import from about 6
seconds to 1 in a sandbox, and from 19 on Kaggle or 47 on Colab where
the filesystem is cold and network-backed.

This is only safe because the bundle ships `tokenizer.json`, which
defines the entire pipeline: normalizer, pre-tokenizer, post-processor
and decoder. Nothing is reconstructed. If a future checkpoint omits that
file, `load_tokenizer` falls back to transformers rather than rebuilding
a BPE pipeline from vocabulary and merges, because a pipeline rebuilt by
hand can differ subtly and a differing token produces a different image
with nothing to signal it.

Verified identical against transformers, token for token, across eleven
prompts covering non-Latin scripts, emoji with modifiers, literal
special-token text, edge whitespace, embedded newlines and tabs, and
lengths that force truncation. Any change here must re-run that
comparison; property tests cannot substitute for it.

One detail worth knowing: Jinja must be configured with `trim_blocks`
and `lstrip_blocks`, which is how transformers configures its own
environment. Without them every control block in the template leaves a
stray newline, and the rendered prompt differs before tokenization even
begins.

### The tokenizer import can cost more than the model load

`transformers` selects a deep learning backend when first imported, and
on a machine with PyTorch installed it imports the whole stack. Measured
on Colab, loading the tokenizer took 47 seconds, nearly all of it
PyTorch and torch_xla being pulled in behind it, for a tokenizer that is
pure Python and Rust and needs no backend.

`src/tokenization/prompt.py` sets `USE_TORCH=0` before the import. It
must be set before transformers is first imported anywhere in the
process, which is why it sits at the import site rather than in a
configuration object.

That alone did not help: a later run still spent 56 seconds there. The
import is simply expensive on a cold, network-backed filesystem,
whatever backend it selects. So the import now runs on a background
thread during the download instead. Both are waiting rather than
computing, so overlapping them removes the cost entirely rather than
reducing it.

### A stage's wall time is meaningless until compilation is separated out

This is the most important thing the telemetry does, and the reason it
was built. A reported decode of 108 seconds might be 105 of compilation
and 3 of work, or the reverse. Compilation is paid once per output
shape and survives in a persistent cache; execution is paid on every
image. The two call for opposite fixes, and a wall time cannot tell
them apart.

`src/telemetry/compilation.py` listens to the events JAX emits for every
compilation, which carry the function name and per-phase durations, and
every timed stage now reports its split automatically. The profile ends
by stating what a cached repeat should cost.

The three phases mean different things. Trace time grows with how much
Python runs, so an unrolled loop pays it and a scan does not. Lowering
grows with program size. Backend compilation is usually the largest and
is what the persistent cache eliminates on later runs.

Measured on CPU while building this: sampling was 97% compilation, and
autoencoder decode was 97% execution. Two stages of comparable wall
time, needing entirely different work.

### Per-stage compilation attribution must be bounded on both sides

Compilation events are recorded process-wide and read at the end of each
stage. Reading alone is not enough: anything pending when a stage begins
belongs to whatever ran before it. Without discarding at the start, the
recorder is a running total, and a real Colab log showed a stage
reporting 3.795s of compilation against 2.347s of wall time.

`timed_stage` now discards pending events on entry and clamps the
reported compilation to the measured wall time, since the two figures
come from different clocks and can disagree at the margins.

Related wording fix: a stage that hands nothing back to the timer may
still block internally. The label says the timer did not wait, not that
the work was unawaited, because the earlier phrasing appeared in a
Kaggle log beside a stage that did block on its own.

### The compilation cache is silent about whether it worked

A cache that misses behaves exactly like no cache: nothing fails,
nothing warns, the run is simply slow again. `warm_up` therefore
snapshots the cache directory before compiling and reports afterwards
what was reused and what had to be built, inferred from which entries
appeared. XLA exposes no hit counter, but a program that compiled left
a file behind and one that did not, did not.

What an entry contains is compiled machine code, not metadata: one
decode program is about 16 MB compressed, 78 MB expanded, and carries
the paths of the source files it came from. Three things decide whether
it is found again: the traced graph, which includes parameter sharding
so a placement change invalidates entries; the jaxlib version; and the
chip generation and topology.

A miss costs only time. It cannot corrupt anything.

Seeding from a read-only location needs a copy first. `/kaggle/input` is
read-only, and XLA needs somewhere writable for programs it compiles
later, so pointing the cache directly at a dataset would let it read but
never write, and anything not already present would recompile on every
run.

### Timing JAX requires blocking, or it measures nothing

JAX dispatch is asynchronous. A timer around a call measures how long
the call took to queue, not how long the work took. Measured on this
codebase: a matrix multiply taking 186ms reports 0.14ms if the timer
does not wait for its result.

`timed_stage` therefore blocks on whatever a stage produces before
stopping the clock, and marks any stage that handed nothing back as
"dispatch only" so its number is not mistaken for real work. This does
cost the overlap JAX would otherwise get between stages; across a
handful of coarse stages that is a good trade for numbers that can be
trusted.

### Telemetry statistics must promote before reducing

`summarise_values` casts to float32 before computing a mean. Summing
bfloat16 in bfloat16 saturates, and an early version reported an array
of ones as having a mean near zero. That is worse than no summary: it
sends a reader hunting for a bug in the stage being described rather
than in the description.

This is the third time the same class of bug has appeared here, after
group normalization and attention scores. When adding any reduction,
promote first.

### Single-device placement is not sharding

`describe_tree` counts only genuinely split leaves as sharded. An array
sitting on one device reports "single device", which is a distinct and
important state: it is what an unplaced parameter looks like, and
conflating it with a replicated or split one is how the autoencoder came
to run on one chip of an eight-chip pod unnoticed for an entire phase.

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
| Transformer | 3x4 latent, timestep 1.0 | n/a | 2.1e-07 |
| Transformer | 4x4 latent, timestep 0.0 | n/a | 2.6e-09 |

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
- Diffusion transformer, complete: multi-axis rotary, timestep
  embedding, modulation, joint attention, both block types and the
  full model, verified against the reference across latent shapes,
  text lengths and timesteps
- Sampler and generation pipeline, complete: schedule, Euler
  integration, latent packing, prompt caching, and an end-to-end
  generation verified against real autoencoder weights
- Execution layer, complete: scan over blocks, fused sampling loop,
  residency planning, device sharding, persistent compilation cache
- Interfaces, complete: shared input handling, in-notebook controls,
  a Gradio front end, and a notebook runner containing no logic

Every phase below is now complete. What remains is not a phase but a
dependency: **none of this has run on TPU**. Every performance claim in
this repository is arithmetic, and the execution layer's options are
proven output-preserving but entirely unmeasured. The first real run
should establish a baseline before anything further is optimized.

Splash attention is deliberately unimplemented for that reason: it
should wait for a measurement showing attention is the bottleneck.

For reference, the order the work was done in:

1. **VAE**: done. Assembly, real-weight execution and reference parity
   are all complete. What remains is measuring decode cost at the
   target resolutions on actual TPU hardware, which cannot be done in
   a CPU sandbox.
2. **Text encoder**: done. What remains is measuring encode cost on
   real hardware, which cannot be done in a CPU sandbox.
3. **Transformer**: done, except the `lax.scan` refactor, which is a
   performance change and belongs with phase 5. Blocks currently run
   in a Python loop over the stacked parameter axis, which is correct
   but compiles the block body once per block.
4. **Sampling**: done. What remains is an end-to-end parity run
   against the reference at a fixed seed, which needs the full-size
   checkpoint and therefore more memory than a CPU sandbox has.
5. **Performance**: done in structure, unmeasured in effect. Every
   option is implemented and proven output-preserving, but none has
   been benchmarked, because none of them does anything measurable on
   CPU. Splash attention remains unimplemented and should wait for a
   real measurement showing attention is the bottleneck.

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
