from typing import Protocol
from re import escape

from outlines.generate import samplers, regex as rgx_gen
from pygtrie import CharTrie

class Tokenizer(Protocol):
    def __call__(self, s: str) -> dict[str, list[int]]: ...
    def batch_decode(self, ids: list[int]) -> list[str]: ...
    def get_vocab(self) -> dict[str, int]: ...

class TokenBoundaryHealer:
    """
    https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38
    The last prompt token harms generation when it prefixes better candidates in the model's vocab.
    So we trim the prompt tokens and let the model re-write them from its own vocab.
    """
    def __init__(self, model, tokenizer: Tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.vocab_trie = CharTrie(tokenizer.get_vocab())

    def __call__(self, prompt: str) -> str:
        prompt_toks = self.tokenizer.batch_decode(self.tokenizer(prompt)['input_ids'])
        if removed_toks := self.trim_toks(prompt_toks):
            prompt = prompt[: prompt.rindex(removed_toks[0])].rstrip()
            for prefix in removed_toks:
                # toks_to_suppress = [i for i, t in self.vocab.items() if not t.startswith(prefix)]
                # model.greedy_search(generation_config=GenerationConfig(..., suppress_tokens=toks_to_suppress))
                # OR
                # https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.SequenceBiasLogitsProcessor
                # self.vocab_bias[prefix] = 100
                # model.greedy_search(generation_config=GenerationConfig(..., sequence_bias=self.vocab_bias))
                # self.vocab_bias[prefix] = -inf
                # OR
                # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.greedy_search.example
                # logits_processor = SuppressTokensLogitsProcessor
                # model.greedy_search(input_ids, GenerationConfig(..., logits_processor=logits_processor))
                prompt += rgx_gen(
                    self.model, f"{escape(prefix)}.*", max_tokens=1,
                    sampler=samplers.greedy # type: ignore[arg-type]
                )(prompt)
        return prompt
        # NOTE: pattern should be "^{t}.*" but outlines.interegular doesn't support `^`
        # https://discord.com/channels/1182316225284554793/1182317824170020895/1188172732735701123

    def trim_toks(self, prompt_toks: list[str]) -> list[str]:
        p_toks = self.trim_falsy_toks(prompt_toks)
        removed_toks: list[str] = []
        while len(self.vocab_trie.items(prefix=p_toks[-1])) > 1:
            removed_toks.insert(0, p_toks.pop()) # NOTE: async masking of logit per popped token?
        # NOTE: https://github.com/guidance-ai/guidance/blob/5f7fa7f6eef6455e6940fe743c5bfdb557330d0b/guidance/llms/_transformers.py#L412-L423
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
