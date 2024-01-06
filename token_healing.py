from itertools import takewhile

from transformers.generation import GenerationConfig
from torch import Tensor, int64
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.vocab = model, CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode

    def __call__(self, prompt: str) -> str:
        prompt_ids = self.encode(prompt, add_special_tokens=False, return_tensors='pt').cuda()
        # TODO: in tests[1], instead of 1189 'read', `trimmed_tok` is 1274 ' read'  beacause that's how it is in `prompt_ids`
        if alt_tok_ids := self.get_tail_alts(prompt_ids):
            healed_ids = self.regenerate_tokens(prompt_ids, alt_tok_ids)
            prompt = self.decode(healed_ids[0], skip_special_tokens=True)
        return prompt

    def get_tail_alts(self, prompt_ids: Tensor) -> list[list[int]]:
        prompt_toks = [self.decode(t).lstrip() for t in prompt_ids[0]]
        tail_toks_extensions = (
            self.vocab.values(prefix=tail_tok) for tail_tok in reversed(prompt_toks)
        ) # retrieving alternatives for each contiguous tail token
        tail_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]
        return tail_alts

    def regenerate_tokens(self, prompt_ids: Tensor, alt_tok_ids: list[list[int]]) -> Tensor:
        if not (n_alts := len(alt_tok_ids)): raise ValueError("Expecting alternative token ids")
        if n_alts == len(prompt_ids[0]): # start from bos if all prompt tokens have alternatives
            ids = Tensor([[self.model.config.bos_token_id]]).to(int64).cuda()
        else:
            ids = prompt_ids[:, : -n_alts] # trim prompt ids
        trimmed_toks = (e.item() for e in prompt_ids[0][-n_alts: ])
        generation_config = GenerationConfig(
            max_new_tokens=1,
            pad_token_id=self.model.config.pad_token_id,
        ) # TODO: fix empty `ids` case
        for trimmed_tok, tok_alts in zip(trimmed_toks, reversed(alt_tok_ids)):
            sequence_bias = {(tok,): 5.0 for tok in tok_alts}
            sequence_bias[(trimmed_tok,)] += 1.0 # limit aggressive healing e.g. 'http'->'https'
            # NOTE: consider `old_tok = tokenize(sequence_to_date)[-1]` for re-aligning ids
            generation_config.update(sequence_bias=sequence_bias)
            ids = self.model.generate(ids, generation_config=generation_config)
        return ids
