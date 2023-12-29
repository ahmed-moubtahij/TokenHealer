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
        prompt_toks = self.tokenizer.batch_decode(p_tok_ids)
        if tok_stack := self.pop_toks(prompt_toks):
            p = prompt[: prompt.rindex(tok_stack[0])].rstrip()
            prompt_ids = self.tokenizer(p, return_tensors='pt').input_ids.cuda()
            max_length_1 = MaxLengthCriteria(1)
            while tok_stack:
                prefix = tok_stack.popleft()
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

    def pop_toks(self, prompt_toks: list[str]) -> deque[str]:
        p_toks = deque(dropwhile(lambda t: not t, reversed(prompt_toks)))
        popped_tok_stack: deque[str] = deque()
        # TODO: Re-use `self.vocab_trie.values(prefix=prefix)` it in __call__?
        while len(self.vocab_trie.values(prefix=p_toks[0])) > 1:
            popped_tok_stack.appendleft(p_toks.popleft())
        return popped_tok_stack
