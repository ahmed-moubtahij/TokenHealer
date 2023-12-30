# What does this do?

It rectifies the [token boundary bias](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38) in greedy tokenization.
Example: Given a prompt with a partial url ending with `:`. The model might have seen the desired `://` as a single token in training, but seeing just `:` tells it: "the next token is likely not `//`, otherwise the token would've been `://`".
This includes the problem of output sensitivity to prompts ending with whitespace or punctuation.

# What do I need to use this?

Python 3.10^ + `pip install .` should pick up `[tool.poetry.dependencies]` from `pyproject.toml`, which are `transformers` and `pygtrie`.

# How do I use this?
`example.py`.

