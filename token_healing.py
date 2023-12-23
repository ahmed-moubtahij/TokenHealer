# https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38
# Due to greedy tokenization, the last prompt token harms generation when it prefixes better candidates in the model's vocab.
# So we could trim the prompt tokens and let the model re-write them from its own vocab.
from outlines.generate import regex as rgx_gen
from re import escape

def debias_token_boundary(prompt, model, tokenizer, max_candidates=10):
    T = tokenizer; vocab = T.get_vocab()
    prompt_toks = [*map(T.decode, T.encode(prompt))]
    if removed_toks := trim_toks(prompt_toks, max_candidates, vocab):
        prompt = prompt[: prompt.rindex(removed_toks[0])].rstrip()
        for t in removed_toks:
            prompt += rgx_gen(model, f"{escape(t)}.*", max_tokens=1)(prompt)
    return prompt
    # NOTE: pattern should be "^{t}.*" but outlines.interegular doesn't support `^`

def trim_toks(prompt_toks, max_candidates, vocab):
    p_toks, removed_toks = trim_falsy_toks(prompt_toks), []
    def many(lst): return len(lst) > max_candidates
    while many([t for t in vocab if t.startswith(p_toks[-1])]):
        removed_toks.insert(0, p_toks.pop())
    return removed_toks

def trim_falsy_toks(prompt_toks):
    r_prompt_toks = prompt_toks[::-1]
    last_truthy_tok = next(filter(None, r_prompt_toks))
    p_toks = prompt_toks[: -r_prompt_toks.index(last_truthy_tok) or None]
    return p_toks

# TODO: repetitive function args -> make this into a `TokenHealer` class
# TODO: use `candidate_tokens` to cache the vocab indices to mask logits with.
# see if the mode's generatefunction has a logits_mask function, otherwise check outlines impl
# TODO: add type hinting
# TODO: Can the outlines logic be inlined and the dependency removed?