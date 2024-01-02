from itertools import takewhile
from transformers.generation import PrefixConstrainedLogitsProcessor, MaxLengthCriteria
from torch import IntTensor
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.vocab_trie = CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.batch_decode = tokenizer.batch_decode

    def __call__(self, prompt: str) -> str:
        trimmed_prompt_ids, toks_alts = self.trim_prompt(prompt)
        if not toks_alts[0]: return prompt
        max_length_1 = MaxLengthCriteria(1)
        def logits_rule(f): return PrefixConstrainedLogitsProcessor(f, num_beams=1)
        for tok_alts in reversed(toks_alts): # regenerate last trimmed toks first
            trimmed_prompt_ids = self.model.greedy_search(
                trimmed_prompt_ids,
                logits_processor=logits_rule(lambda *_, allowed_toks=tok_alts: allowed_toks),
                stopping_criteria=max_length_1,
                pad_token_id=self.model.config.pad_token_id,
                # use_cache=True,
            )
        healed_prompt = self.decode(trimmed_prompt_ids.squeeze(), skip_special_tokens=True)
        return healed_prompt

    def trim_prompt(self, prompt: str) -> tuple[IntTensor, list[list[int]]]:
        prompt_ids = self.encode(prompt, return_tensors='pt').cuda()
        prompt_toks = self.batch_decode(prompt_ids.squeeze())

        tail_toks_extensions = ( # ids of e.g. ['.', ':'] -> [['.', '. '], [':', '://']]
            self.vocab_trie.values(prefix=tail_tok.lstrip()) for tail_tok in reversed(prompt_toks)
        ) # querying contiguous tail tokens for alternative tokens
        trimmed_toks_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]

        return prompt_ids[:, : -len(trimmed_toks_alts) or None], trimmed_toks_alts
