from itertools import takewhile

from transformers.generation import GenerationConfig
from torch import Tensor, int64
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.vocab = model, CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.space_tok = tokenizer.tokenize(' ')[0]
        self.bos_id, self.pad_id = tokenizer.bos_token_id, tokenizer.pad_token_id

    def __call__(self, prompt: str) -> str:
        prompt_ids = self.encode(prompt, add_special_tokens=False)
        if alt_tok_ids := self.get_tail_alts(prompt_ids):
            healed_ids = self.regenerate_tokens(prompt_ids, alt_tok_ids)
            prompt = self.decode(healed_ids[0], skip_special_tokens=True)
        return prompt

    def get_tail_alts(self, prompt_ids: list[int]) -> list[list[int]]:
        prompt_toks = [self.decode(t) for t in prompt_ids]
        tail_toks_extensions = (
            self.vocab.values(prefix=tail_tok.replace(' ', self.space_tok))
            for tail_tok in reversed(prompt_toks)
        ) # retrieving alternatives for each contiguous tail token
        tail_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]
        return tail_alts

    def regenerate_tokens(self, prompt_ids: list[int], alt_tok_ids: list[list[int]]) -> Tensor:
        if not (n_alts := len(alt_tok_ids)): raise ValueError("Expecting alternative token ids")
        # trim prompt if not all tokens have alternatives, otherwise regenerate from bos
        ids_ = prompt_ids[: -n_alts] if n_alts < len(prompt_ids) else [self.bos_id]
        ids = Tensor([ids_]).to(int64).cuda()
        gen_cfg = GenerationConfig(max_new_tokens=1, pad_token_id=self.pad_id)
        for trimmed_tok, tok_alts in zip(prompt_ids[-n_alts: ], reversed(alt_tok_ids)):
            sequence_bias = {(tok,): 10.0 for tok in tok_alts}
            sequence_bias[(trimmed_tok,)] += 1.0 # limit aggressive healing e.g. 'http'->'https'
            gen_cfg.update(sequence_bias=sequence_bias)
            ids = self.model.generate(ids, generation_config=gen_cfg)
        return ids
