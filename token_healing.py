from transformers.generation import GenerationConfig
from torch import Tensor, int64
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        t, self.model = tokenizer, model
        self.vocab, self.space_tok = CharTrie(t.get_vocab()), t.tokenize(' ')[0]
        self.encode, self.decode = t.encode, t.decode
        self.gen_cfg = GenerationConfig(
            max_new_tokens=1, bos_token_id=t.bos_token_id, pad_token_id=t.pad_token_id
        )

    def __call__(self, prompt: str) -> str:
        prompt_ids = self.encode(prompt, add_special_tokens=False)

        tail_tok = self.decode(prompt_ids[-1]).replace(' ', self.space_tok)
        seq_bias = {(alt_tok,): 10.0 for alt_tok in self.vocab.values(prefix=tail_tok)}
        if not seq_bias: return prompt

        seq_bias[(prompt_ids[-1],)] += 1.0 # limit aggressive healing e.g. 'http'->'https'
        ids = Tensor([prompt_ids[: -1]]).to(int64).cuda() if len(prompt_ids) > 1 else None
        self.gen_cfg.update(sequence_bias=seq_bias)

        healed_ids = self.model.generate(ids, generation_config=self.gen_cfg)
        healed_prompt = self.decode(healed_ids.squeeze(), skip_special_tokens=True)
        return healed_prompt
