# Conversion notes

The pipeline that produced the checkpoint bundle lives outside this
repository, but one decision it made is worth recording here, because
it caused a bug that took a long time to surface.

## Copy the whole tokenizer directory, never a list of names

The original converter named the tokenizer files it wanted:

```python
tokenizer_filenames = (
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
)
```

Six files. The upstream repository has seven. The missing one was
`tokenizer.json`, which is both the largest and the only one that
defines the tokenization pipeline completely.

The list was assembled from what the converter's own loader needed at
the time, on the assumption that `AutoTokenizer` would rebuild the
pipeline from vocabulary and merges. That assumption held, so nothing
failed and nobody noticed. It only surfaced months later, when a faster
tokenizer that needs `tokenizer.json` could not be used at all, on a
bundle whose source had shipped that file from the beginning.

Two lessons, and the second matters more:

The whole directory is 15.9 MB against a bundle of 11 GB. Selecting
files to save 11 MB, at the cost of possibly omitting an important one,
is a bad trade at any ratio and an absurd one at this ratio.

More importantly, a hand-written list encodes today's implementation
choices into the artifact. Whoever wants a different loader later finds
the bundle already decided against them. Copy the directory as it
stands; the upstream repository is the authority on what belongs in it.

The same reasoning applies to any future directory the bundle carries.
