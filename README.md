# What does this do?

It rectifies the [token boundary bias](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38) of tokenizers.
Example: Given a prompt with a partial url ending with `:`. The model might have seen the desired `://` as a single token in training, but seeing just `:` tells it "the next token is likely not `//`, otherwise the token would've been `://`".
It also addresses the problem of output sensitivity to prompts ending with punctuation or whitespace.

# How do I use this?

[outlines](https://github.com/outlines-dev/outlines) is the only dependency, so Just `pip install outlines` in your project and copy-paste the code in `token_healing.py`. Look at `example.py` for how to use it.
