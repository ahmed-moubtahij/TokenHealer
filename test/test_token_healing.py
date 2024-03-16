import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer
from parameterized import parameterized
from tokenhealing import TokenBoundaryHealer

class TokenHealingTestCase(unittest.TestCase):

    def setUp(self):
        model_name_or_path = 'TheBloke/deepseek-llm-7B-base-GPTQ'
        self.completion_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            trust_remote_code=False,
            revision='main',
            use_cache=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.token_healer = TokenBoundaryHealer(self.completion_model, self.tokenizer)

    @parameterized.expand(
        [
            (
                "square_bracket",
                'An example ["like this"] and another example [',
                'An example ["like this"] and another example ["',
            ),
            ("url", 'The link is <a href="http:', 'The link is <a href="http://'),
            ("aggressive_healing", 'The link is <a href="http', 'The link is <a href="http'),
            ("nothing_to_heal", "I read a book about", "I read a book about"),
            ("single_token", "I", "I"),
            ("empty_prompt", "", ""),
        ]
    )
    def test_prompts(self, test_name, input, expected):
        healed_prompt = self.token_healer(input)
        self.assertEqual(healed_prompt, expected, f"Failed test: {test_name}")

if __name__ == '__main__':
    unittest.main()
