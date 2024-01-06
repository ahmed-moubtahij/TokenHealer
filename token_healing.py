from itertools import takewhile

from transformers.generation import MaxLengthCriteria
from torch import IntTensor
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.vocab = tokenizer.get_vocab()
        self.vocab_trie = CharTrie(self.vocab)
        self.model = model
        self.tokenizer = tokenizer
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.batch_decode = tokenizer.batch_decode
        self.max_length_1 = MaxLengthCriteria(1)
        # NOTE: why is self.model.config.vocab_size != tokenizer.vocab_size

    def __call__(self, prompt: str) -> str:
        prompt_ids = self.encode(prompt, add_special_tokens=False, return_tensors='pt').cuda()
        prompt_toks = [self.decode(t).lstrip() for t in prompt_ids.squeeze()]
        if alts := self.get_tail_alts(prompt_toks):
            healed_ids = self.regenerate_tokens(prompt_ids, alts)
            prompt = self.decode(healed_ids.squeeze(), skip_special_tokens=True)
        return prompt

    # def get_tail_alts(self, prompt_ids: IntTensor) -> list[list[int]]:
    def get_tail_alts(self, prompt_toks: list[str]) -> list[list[int]]:
        tail_toks_extensions = (
            self.vocab_trie.values(prefix=tail_tok.lstrip())
            for tail_tok in reversed(prompt_toks)
        ) # retrieving alternatives for each contiguous tail token
        tail_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]
        return tail_alts

        ids = prompt_ids[:, : -len(alt_tok_ids)] # trim prompt ids
                pad_token_id=self.model.config.pad_token_id,
                # normalize_logits=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            ids = ids.sequences
            # # Regenerating pre-existing substrings might be useful for aligning token ids
            # if not (new_tok := self.decode(ids[0][-1])) in prompt:
            #     # Regenerating a non-pre-existing substring; need to preserve prompt integrity
            #     # This is where you somehow preserve `http` over `https`
            #     # scores = ids.scores[0].squeeze() # idk what to do with this yet
            #     pass
            # ids = ids.sequences
        return ids
