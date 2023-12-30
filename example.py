from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from token_healing import TokenBoundaryHealer

def generate(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    generation_config = GenerationConfig(
        temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=32,
        pad_token_id=model.config.pad_token_id,
    )
    output = model.generate(inputs=input_ids, generation_config=generation_config)
    return tokenizer.decode(output[0], skip_special_tokens=True)

model_name_or_path = 'TheBloke/deepseek-llm-7B-base-GPTQ'
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map='auto',
    trust_remote_code=False,
    revision='main'
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# test queries from
# https://github.com/guidance-ai/guidance/blob/5f7fa7f6eef6455e6940fe743c5bfdb557330d0b/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb
test_prompts = [
    'The link is <a href="http:',
    'what is the DNS address ?! ',
    'I read a book about ',
    'I read a book about',
    'An example ["like this"] and another example ['
]
prompt = test_prompts[0]
print(f'\nOriginal prompt:\n{prompt}\n')

unguided_output = generate(prompt, model, tokenizer)
print(f'Generation with original prompt:\n{unguided_output}\n')

token_healer = TokenBoundaryHealer(model, tokenizer)
healed_prompt = token_healer(prompt)
print(f'Healed prompt:\n{healed_prompt}\n')
guided_output = generate(healed_prompt, model, tokenizer)
print(f'Generation with healed prompt:\n{guided_output}\n\n')
