import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel


model_path = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype = torch.float16,
    device_map = {"": 0},
)

# Load LoRA and merge
model = PeftModel.from_pretrained(model, "adityasihag/math_QA-Mistral-7B-QLoRA-adapter")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

question = """Solve the linear equations. $3(x+2)-x=x + 9$"""

sample_input = f"""Question: {question}. Find the value of x. \n Answer: Let's think step by step. """

sample_input_tokenised = tokenizer(sample_input, return_tensors = "pt").to("cuda")

generated_ids = model.generate(
                    **sample_input_tokenised,
                    max_new_tokens = 1024,
                    temperature = 0.3
                )
output = tokenizer.decode(generated_ids[0], skip_special_tokens = True)
print(output)
