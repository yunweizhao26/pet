import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 128
model_name = 'facebook/opt-66b'
#model_name = 'EleutherAI/gpt-neox-20b'
#text = """Hello, I am a prompt. Who are you?"""
text = 'Hamburg is in which country?\n'
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids
free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f'{free_in_GB-2}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map='auto',
  load_in_8bit=True,
  max_memory=max_memory
)
generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

