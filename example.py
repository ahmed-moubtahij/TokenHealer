import outlines
from token_healing import TokenHealer

model = outlines.models.gptq(
    "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ",
    device="auto",
    model_kwargs={"trust_remote_code": False, "revision": "main"},
    tokenizer_kwargs={"use_fast": True}
)
token_healer = TokenHealer(model, model.tokenizer.tokenizer)
MAX_TOKENS = 64
prompt_template="""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
system_message = "You are a helpful assistant."

# TODO: find better demonstrative prompts
query = "What is the IP address of the Google DNS servers ? ! "
print(f"\nQUERY:\n{query}\n")
prompt = prompt_template.format(system_message=system_message, query=query)
unguided = outlines.generate.text(model, max_tokens=MAX_TOKENS)(prompt)
print(f"UNGUIDED:\n{unguided}\n\n")

healed_query = token_healer(query)
print(f"HEALED QUERY:\n{healed_query}\n")
healed_prompt = prompt_template.format(system_message=system_message, query=healed_query)
guided = outlines.generate.text(model, max_tokens=MAX_TOKENS)(healed_prompt)
print(f"GUIDED:\n{guided}\n")
