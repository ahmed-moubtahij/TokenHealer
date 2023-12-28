from collections import deque
from itertools import dropwhile
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
        input_ids = self.tokenizer(prompt)['input_ids']
        prompt_toks = self.tokenizer.batch_decode(input_ids)
        if tok_stack := self.pop_toks(prompt_toks):
            prompt = prompt[: prompt.rindex(tok_stack[0])].rstrip()
            while tok_stack:
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
                prefix = f"{escape(tok_stack.popleft())}.*"
                prompt += rgx_gen(
                    self.model, prefix, max_tokens=1,
                    sampler=samplers.greedy # type: ignore[arg-type]
                )(prompt)
        return prompt
        # NOTE: pattern should be "^{t}.*" but outlines.interegular doesn't support `^`
        # https://discord.com/channels/1182316225284554793/1182317824170020895/1188172732735701123

    def pop_toks(self, prompt_toks: list[str]) -> deque[str]:
        p_toks = deque(dropwhile(lambda t: not t, reversed(prompt_toks)))
        popped_tok_stack: deque[str] = deque()
        while len(self.vocab_trie.items(prefix=p_toks[0])) > 1:
            popped_tok_stack.appendleft(p_toks.popleft()) # NOTE: async logits mask per popped token
        # NOTE: https://github.com/guidance-ai/guidance/blob/5f7fa7f6eef6455e6940fe743c5bfdb557330d0b/guidance/llms/_transformers.py#L412-L423
        return popped_tok_stack

# TODO: check if `GenerationConfig` params can help control generation instead of using outlines
# https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.suppress_tokens
# TODO: use `candidate_tokens` to cache the vocab indices to mask logits with.
# TODO: see if the model's generate function has a logits_mask function,
# TODO: otherwise check outlines impl.
# TODO: Can the outlines logic be inlined and the dependency removed?
# TODO: Important because you're forcing the user to wrap their model in it
