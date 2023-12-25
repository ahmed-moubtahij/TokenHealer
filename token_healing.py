# https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38
# Due to greedy tokenization, the last prompt token harms generation when it prefixes better candidates in the model's vocab.
# So we could trim the prompt tokens and let the model re-write them from its own vocab.
from outlines.generate import regex as rgx_gen
from re import escape

# SPIECE_UNDERLINE = "▁" # http://tinyurl.com/5xrpm655

class TokenHealer:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.vocab = tokenizer.get_vocab()

    def __call__(self, prompt):
        prompt_toks = [*map(self.tokenizer.decode, self.tokenizer(prompt)['input_ids'])]
        if removed_toks := self.trim_toks(prompt_toks):
            tail_idx = prompt.rindex(removed_toks[0])
            prompt = prompt[: tail_idx].rstrip()
            for t in removed_toks:
                prompt += rgx_gen(self.model, f"{escape(t)}.*", max_tokens=1)(prompt)
        return prompt
        # NOTE: pattern should be "^{t}.*" but outlines.interegular doesn't support `^`
        # https://discord.com/channels/1182316225284554793/1182317824170020895/1188172732735701123

    def trim_toks(self, prompt_toks):
        p_toks, removed_toks = self.trim_falsy_toks(prompt_toks), []
        while len([t for t in self.vocab if t.startswith(p_toks[-1])]) > 1:
            removed_toks.insert(0, p_toks.pop())
        return removed_toks

    def trim_falsy_toks(self, prompt_toks):
        truthy_idx = next(i for i, t in enumerate(reversed(prompt_toks)) if t)
        return prompt_toks[: -truthy_idx or None]

# TODO: add type hinting
# TODO: use `candidate_tokens` to cache the vocab indices to mask logits with.
# see if the mode's generatefunction has a logits_mask function, otherwise check outlines impl
# TODO: Can the outlines logic be inlined and the dependency removed?