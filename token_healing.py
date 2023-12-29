from collections import deque
from itertools import dropwhile

from transformers.generation import PrefixConstrainedLogitsProcessor, MaxLengthCriteria
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.vocab_trie = CharTrie(tokenizer.get_vocab())

    def __call__(self, prompt: str) -> str:
        p_tok_ids = self.tokenizer(prompt)['input_ids']
        dead_toks, trimmed_prompt, extension_options = self.trim_prompt(p_tok_ids)
        if dead_toks:
            prompt_ids = self.tokenizer(
                trimmed_prompt, return_tensors='pt'
            ).input_ids.cuda()
            max_length_1 = MaxLengthCriteria(1)
            while dead_toks:
                prefix = dead_toks.popleft()
                logits_processor = PrefixConstrainedLogitsProcessor(
                    lambda *_: self.vocab_trie.values(prefix=prefix), #pyright: ignore
                    num_beams=1,
                )
                prompt_ids = self.model.greedy_search(
                    prompt_ids,
                    logits_processor=logits_processor,
                    stopping_criteria=max_length_1,
                    pad_token_id=self.model.config.pad_token_id,
                )
            prompt = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)[0]
        return prompt

    def trim_prompt(self, tok_ids: list[int]) -> tuple[deque[str], str, deque[list[str]]]:
        prompt_toks = self.tokenizer.batch_decode(tok_ids)
        truthy_toks = dropwhile(lambda t: not t, reversed(prompt_toks))
        p_toks = deque(truthy_toks); p_toks.reverse()

        dead_toks: deque[str] = deque()
        extension_options: deque[list[str]] = deque()
        # TODO: Re-use `candidates` it in __call__
        while len(options := self.vocab_trie.keys(prefix=p_toks[-1])) > 1:
            dead_toks.appendleft(p_toks.pop())
            extension_options.appendleft(options)

        trimmed_prompt = self.tokenizer.decode(tok_ids[: -len(dead_toks)])
        return dead_toks, trimmed_prompt, extension_options
