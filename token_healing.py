# https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38
# Due to greedy tokenization, the last prompt token harms generation when
# it prefixes better candidates in the model's vocab. So we could trim
# the prompt tokens and let the model re-write them from its own vocab.
from typing import Protocol
from re import escape
from outlines.generate import samplers, regex as rgx_gen

class Tokenizer(Protocol):
    def __call__(self, s: str) -> dict[str, list[int]]: ...
    def batch_decode(self, ids: list[int]) -> list[str]: ...
    def get_vocab(self) -> dict[str, int]: ...

class TokenHealer:
    def __init__(self, model, tokenizer: Tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.vocab = tokenizer.get_vocab()

    def __call__(self, prompt: str) -> str:
        prompt_toks = self.tokenizer.batch_decode(self.tokenizer(prompt).input_ids)
        if removed_toks := self.trim_toks(prompt_toks):
            prompt = prompt[: prompt.rindex(removed_toks[0])].rstrip()
            for t in removed_toks:
                prompt += rgx_gen(
                    self.model, f"{escape(t)}.*", max_tokens=1,
                    sampler=samplers.greedy # type: ignore[arg-type]
                )(prompt)
        return prompt
        # NOTE: pattern should be "^{t}.*" but outlines.interegular doesn't support `^`
        # https://discord.com/channels/1182316225284554793/1182317824170020895/1188172732735701123

    def trim_toks(self, prompt_toks: list[str]) -> list[str]:
        p_toks = self.trim_falsy_toks(prompt_toks)
        removed_toks: list[str] = []
        while len([t for t in self.vocab if t.startswith(p_toks[-1])]) > 1:
            removed_toks.insert(0, p_toks.pop()) # NOTE: async mask logits for popped token?
        return removed_toks

    def trim_falsy_toks(self, prompt_toks: list[str]) -> list[str]:
        truthy_idx = next(i for i, t in enumerate(reversed(prompt_toks)) if t)
        return prompt_toks[: -truthy_idx or None]

# TODO: check if `GenerationConfig` params can help control generation instead of using outlines
# https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.suppress_tokens
# TODO: use `candidate_tokens` to cache the vocab indices to mask logits with.
# TODO: see if the model's generate function has a logits_mask function,
# TODO: otherwise check outlines impl.
# TODO: Can the outlines logic be inlined and the dependency removed?
# TODO: Important because you're forcing the user to wrap their model in it
