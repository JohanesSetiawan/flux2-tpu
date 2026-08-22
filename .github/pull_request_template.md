## What this changes

<!-- One or two sentences. The diff says how; say why. -->

## Verification

- [ ] `JAX_ENABLE_X64=1 python -m tests.run_all_tests` passes
- [ ] Tested on TPU, or noted below that it was not

<!--
CI runs the unit suite on CPU, which covers logic but cannot see
allocation failures, compilation cost, or precision-mode behaviour.
All three have reached production here despite a passing suite, so
saying whether you tested on hardware is genuinely useful.
-->

## If this changes numerics

<!--
Delete this section if it does not.

Most bugs in this codebase produce correctly shaped output containing
different numbers, with nothing raised. If you changed anything that
affects what is computed, say which parity test covers it, or add one.
AGENTS.md sections 3 and 4 list the conventions that look arbitrary and
are not.
-->
