from itertools import takewhile

from transformers.generation import MaxLengthCriteria
from torch import IntTensor
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.vocab_trie = model, CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.max_length_1 = MaxLengthCriteria(1)

    def __call__(self, prompt: str) -> str:
        prompt_ids = self.encode(prompt, return_tensors='pt').cuda()
        if alts := self.get_tail_alts(prompt_ids):
            trimmed_ids = prompt_ids[:, : -len(alts) or None]
            healed_ids = self.regenerate_tokens(trimmed_ids, alts)
            prompt = self.decode(healed_ids.squeeze(), skip_special_tokens=True)
        return prompt

    def get_tail_alts(self, prompt_ids: IntTensor) -> list[list[int]]:
        prompt_toks = [*map(self.decode, prompt_ids.squeeze())]
        tail_toks_extensions = (
            self.vocab_trie.values(prefix=tail_tok.lstrip())
            for tail_tok in reversed(prompt_toks)
        ) # retrieving alternatives for each contiguous tail token
        tail_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]
        return tail_alts

    def regenerate_tokens(self, ids: IntTensor, tail_alts: list[list[int]]) -> IntTensor:
        for tok_alts in reversed(tail_alts): # regenerate last trimmed toks first
            ids = self.model.greedy_search(
                ids,
                force_words_ids=[[tok_alts]],
                stopping_criteria=self.max_length_1,
                pad_token_id=self.model.config.pad_token_id,
            )
        return ids
