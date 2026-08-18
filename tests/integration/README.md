# Integration tests

These tests are deliberately separate from the unit suite run by
`python -m tests.run_all_tests`, because they have requirements the
unit suite does not: network access, multi-gigabyte downloads, and in
some cases PyTorch.

PyTorch is used here **only** to produce reference outputs to compare
against. It is not, and must never become, a dependency of the
`flux2_klein` package itself.

Run them individually and explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install einops safetensors

python -m tests.integration.test_vae_parity
```
