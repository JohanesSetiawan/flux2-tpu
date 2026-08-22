# Security

## Reporting a vulnerability

Open a [security advisory](https://github.com/JohanesSetiawan/flux2-tpu/security/advisories/new)
rather than a public issue, and expect a slower response than a staffed
project; this is a side project.

## What this software does not protect against

Worth stating directly, because it is easy to assume otherwise from a
project that otherwise looks finished.

**There is no content filtering.** No prompt filtering, no output
classification, nothing. The reference implementation ships NSFW and
protected-content filters for both inputs and outputs, which Black
Forest Labs requires for the 9B checkpoints and encourages for 4B. None
of that is ported here.

**There is no watermarking.** The reference implements pixel-layer
watermarking and documents C2PA metadata so generated images can be
identified as synthetic. Images this produces carry no marker of any
kind.

Neither omission is a judgement that they are unnecessary. They were out
of scope for a port aimed at numerical parity, and adding either is
separate work rather than a setting.

If you deploy this where other people can reach it, you are deploying a
generative image model with no safeguards, and the responsibility for
that is yours. The checkpoint's own safety training remains intact, but
that is a property of the weights rather than of this code, and it is
not a substitute for a filter.

## Running untrusted input

The generation path takes a prompt string and turns it into token
identifiers; it does not evaluate anything. The web interface is Gradio,
and `launch(share=True)` creates a **public** link that anyone with the
URL can use while the cell runs. That is convenient in a hosted notebook
and worth thinking about anywhere else.

## Weights

Weights are not distributed here. They come from
[Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
and are loaded from a checkpoint you supply. Loading a checkpoint means
executing whatever Orbax and JAX do with it, so treat an untrusted
checkpoint the way you would treat any untrusted file.
