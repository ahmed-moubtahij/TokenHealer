from collections import deque
from itertools import dropwhile
from typing import Protocol

from transformers.generation import SuppressTokensLogitsProcessor, MaxLengthCriteria
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
            trimmed_prompt = prompt[: prompt.rindex(tok_stack[0])].rstrip()
            vocab_ids = set(self.vocab_trie.itervalues())
            prompt_ids = self.tokenizer(
                trimmed_prompt,
                return_tensors='pt', # type: ignore
            )['input_ids'].to('cuda:0') # type: ignore
            max_length_1 = MaxLengthCriteria(1)
            while tok_stack:
                prefix = tok_stack.popleft()
                toks_to_suppress = vocab_ids - set(self.vocab_trie.itervalues(prefix=prefix))
                prompt_ids = self.model.greedy_search(
                    prompt_ids,
                    logits_processor=SuppressTokensLogitsProcessor(toks_to_suppress),
                    stopping_criteria=max_length_1,
                    pad_token_id=self.model.config.pad_token_id,
                )
                # https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.SequenceBiasLogitsProcessor
                # self.vocab_bias[prefix] = 100
                # self.model.greedy_search(sequence_bias=self.vocab_bias)
                # self.vocab_bias[prefix] = -inf
            prompt = self.tokenizer.batch_decode(
                prompt_ids,
                skip_special_tokens=True, # type: ignore
            )[0]
        return prompt

    def pop_toks(self, prompt_toks: list[str]) -> deque[str]:
        p_toks = deque(dropwhile(lambda t: not t, reversed(prompt_toks)))
        popped_tok_stack: deque[str] = deque()
        while len(self.vocab_trie.keys(prefix=p_toks[0])) > 1:
            popped_tok_stack.appendleft(p_toks.popleft()) # TODO: async logits mask per popped token
        # Draw inspiration from torch.scatter:
        # https://github.com/guidance-ai/guidance/blob/
        # 5f7fa7f6eef6455e6940fe743c5bfdb557330d0b/guidance/llms/_transformers.py#L412-L423
        return popped_tok_stack
