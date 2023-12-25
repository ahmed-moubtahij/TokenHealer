import outlines
from token_healing import TokenHealer

model = outlines.models.awq("TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-AWQ")
token_healer = TokenHealer(model, model.tokenizer.tokenizer)
# max_tokens = 50
# TODO: find better demonstrative prompts
prompt = "What is the IP address of the Google DNS servers ? ! "
# unguided = outlines.generate.text(model, max_tokens=max_tokens)(prompt)
# print(f"\nUNGUIDED:\n{unguided}\n")

healed_prompt = token_healer(prompt)
print(f"\n{prompt=}\n{healed_prompt=}\n")
# guided = outlines.generate.text(model, max_tokens=max_tokens)(healed_prompt)

# print(f"\nGUIDED:\n{guided}\n")
