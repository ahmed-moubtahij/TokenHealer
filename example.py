import outlines
from token_healing import debias_token_boundary

model = outlines.models.awq("TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-AWQ")
# max_tokens = 50
# TODO: find a better demonstrative prompt
prompt = "What is the IP address of the Google DNS servers ? ! "
# unguided = outlines.generate.text(model, max_tokens=max_tokens)(prompt)
# print(f"\nUNGUIDED:\n{unguided}\n")

healed_prompt = debias_token_boundary(prompt, model, model.tokenizer.tokenizer)
print(f"\n{prompt=}\n{healed_prompt=}\n")
# guided = outlines.generate.text(model, max_tokens=max_tokens)(healed_prompt)

# print(f"\nGUIDED:\n{guided}\n")
