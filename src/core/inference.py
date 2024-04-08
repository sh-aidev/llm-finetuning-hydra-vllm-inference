import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    "outputs",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("outputs")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

INPUT = """
### Instruction:
List 3 historical events related to the following country

### Input:
Canada

### Response:
"""

input_ids = tokenizer(INPUT, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = tokenizer.decode(
    model.generate(input_ids=input_ids, max_new_tokens=400, do_sample=True, top_p=0.9, temperature=0.9)[0],
    skip_special_tokens=True
)
print(outputs)
