# Integration tests

These tests are deliberately separate from the unit suite run by
`python -m tests.run_all_tests`, because they have requirements the
unit suite does not: network access, multi-gigabyte downloads, and in
some cases PyTorch.

PyTorch is used here **only** to produce reference outputs to compare
against. It is not, and must never become, a dependency of the
`src` package itself.

Run them individually and explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install einops safetensors

python -m tests.integration.test_vae_parity
```

Available tests:

```bash
# Numerical parity of the VAE decoder against the reference PyTorch
# implementation. Needs a checkout of the reference source.
git clone https://github.com/black-forest-labs/flux2 /tmp/flux2
python -m tests.integration.test_vae_parity --reference-source-path /tmp/flux2

# Numerical parity of the text encoder against reference transformers
# Qwen3, swept across padding levels.
python -m tests.integration.test_text_encoder_parity

# Structural contract between the code and the real checkpoint bundle.
# Reads metadata only, so it does not need memory proportional to the
# checkpoint size.
python -m tests.integration.test_checkpoint_structure
```
