from transformers.generation import GenerationConfig
from torch import Tensor, int64

from tokenhealing.trie import Trie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        t, self.model = tokenizer, model
        self.vocab, self.space_tok = Trie(t.get_vocab()), t.tokenize(' ')[0]
        self.encode, self.decode = t.encode, t.decode
        self.gen_cfg = GenerationConfig(
            max_new_tokens=1, bos_token_id=t.bos_token_id, pad_token_id=t.pad_token_id,
        )

    def __call__(self, prompt: str) -> str:
        if not prompt: return prompt
        # assumption: leading/trailing whitespace is not meaningful, so the prompt is
        # stripped before encoding to desensitize generation to whitespace artefacts
        prompt_ids = self.encode(prompt.strip(), add_special_tokens=False)

        # tail token is used for a prefix search, thus, whitespaces are replaced with their
        # tokenization (e.g. 'Ġ') to enable search for tokens prefixed with a whitespace
        tail_tok = self.decode(prompt_ids[-1]).replace(' ', self.space_tok)

        # apply bias on token id alternatives, i.e., extensions of the tail token
        seq_bias = {(alt_tok,): 10.0 for alt_tok in self.vocab.extensions(tail_tok)}
        if len(seq_bias) == 1:
            return prompt # skip if there are no token alternatives to heal with

        # slightly favor original token to limit aggressive healing e.g. 'http' -> 'https'
        seq_bias[(prompt_ids[-1],)] += 1.0
        self.gen_cfg.update(sequence_bias=seq_bias)

        if len(prompt_ids) > 1: trimmed_ids = Tensor([prompt_ids[: -1]]).to(int64).cuda()
        else:                   trimmed_ids = None # prompt is a single token -> regen from bos

        healed_ids = self.model.generate(trimmed_ids, generation_config=self.gen_cfg)
        healed_prompt = self.decode(healed_ids.squeeze(), skip_special_tokens=True)
        return healed_prompt
