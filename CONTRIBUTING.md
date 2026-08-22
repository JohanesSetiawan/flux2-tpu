# Contributing

Thanks for looking. This is a research side project rather than a
maintained product, so expect a slower response than a staffed
repository, but issues and pull requests are genuinely welcome.

## Before you change anything

Read [AGENTS.md](AGENTS.md). It is not a style guide; it records the
decisions behind the code and the mistakes already made, several of
which look arbitrary without the reasoning attached.

The single most useful thing to understand first: **in this codebase,
most bugs do not raise.** A wrong rotary convention, a mismatched
dtype, a mis-selected hidden state, a subtly different tokenization all
produce correctly shaped output containing different numbers. Nothing
fails, and the only symptom is a different image.

That is why the conventions are pinned by tests rather than by comments,
and why a change that "looks equivalent" needs a test showing it is.

## Setting up

```bash
git clone https://github.com/JohanesSetiawan/flux2-tpu.git
cd flux2-tpu
pip install -e .
JAX_ENABLE_X64=1 python -m tests.run_all_tests
```

The 64-bit flag is not optional. The regression suites compare against
float64 oracles, and without it they compare float32 against float32 and
prove much less.

No TPU is needed for the unit suite. Sharding is exercised across
simulated devices, so placement logic is testable on CPU.

## Tests

Three kinds, and a change usually needs the same kind as the code it
touches:

**Smoke** tests check shape, dtype, and that the thing runs. Cheap, and
they catch wiring mistakes.

**Regression** tests check numerical correctness against an
independently implemented oracle, or against a property that must hold
regardless of the values. Inputs are generated at run time from seeded
shapes. Do not add golden arrays: a stored expected value cannot tell
you whether it was right when it was recorded.

**Precision** tests run at bfloat16, which is what production uses. A
suite running entirely in float64 cannot see a dtype promotion, and one
reached production that way.

Some properties cannot be checked by an oracle, because an oracle
sharing the implementation's assumption would agree with it. Those are
tested structurally instead: placing a unit value in one feature and
checking which index receives the rotated component, for example. When
you find yourself unable to write an oracle, that is usually the reason,
and a structural test is the answer.

## Integration tests

Five tests in [tests/integration/](tests/integration/) compare against
the reference implementations this code was ported from. They need
network access and, for parity, PyTorch, so they are excluded from the
unit suite and run explicitly. If you change anything numeric, run the
relevant one.

## What is likely to be rejected

Not because the idea is bad, but because these have specific reasons
recorded in AGENTS.md:

- Changing the sampler's step count, guidance, or integration method.
  The checkpoint was distilled against one specific four-step Euler
  trajectory; a "better" solver is worse at equal cost.
- Merging the two rotary implementations. They use incompatible pairing
  conventions and each matches only its own checkpoint.
- Adding a mask to the transformer's attention. It is unmasked in the
  reference, deliberately.
- Optimisations without a measurement. Several plausible ones in this
  repository's history turned out to be neutral or worse on hardware,
  including one that a reasonable argument predicted would halve compile
  time and instead increased it.

## Reporting a problem

The run log is usually enough to diagnose without reproducing anything.
It carries a per-stage timing breakdown separating compilation from
execution, tensor placement per device, and whether the compilation
cache was reused. Please include it.
